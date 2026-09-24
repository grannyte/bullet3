/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
//
// Plain TCP rather than the vendored ENet: the tick barrier IS a blocking recv on a 1:1 reliable
// ordered stream, and none of ENet's channels/unreliable/multi-peer machinery is used here.

#include "DemoNetSync.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#pragma comment(lib, "ws2_32.lib")

namespace
{
const unsigned int kNetMagic = 0x534e534fu;   // 'OSNS'
const unsigned int kNetVersion = 1;
const int kRecvTimeoutMs = 300000;
/// A snapshot for 600k bodies is tens of MB; anything past this is a framing error, not a payload.
const unsigned int kMaxPayloadBytes = 256u * 1024u * 1024u;

enum NetMsg
{
	MSG_HELLO = 1,
	MSG_BEGIN,   ///< Tick carries the backend index; payload is {resetOk, canSnapshot}.
	MSG_HASH,    ///< Lockstep: tick + 8-byte state hash.
	MSG_TICK,    ///< Authoritative: tick + 8-byte host hash + optional snapshot bytes.
	MSG_ACK,
	MSG_END,
	MSG_BUCKETS, ///< Healing: tick + one u64 per hash bucket. Empty payload means "cannot bucket".
	MSG_HEAL     ///< Healing: tick + a CaptureBuckets payload, host to client. Empty means nothing.
};

#pragma pack(push, 1)
struct FrameHeader
{
	unsigned int Magic;
	unsigned int Type;
	unsigned long long Tick;
	unsigned int Length;
};

struct HelloPayload
{
	unsigned int Version;
	unsigned int Mode;
	unsigned int Interval;
	unsigned int CorrectEvery;
	unsigned int Steps;
	unsigned int BackendCount;
	unsigned int BodyCount;
	unsigned long long SceneHash;
};

struct AckPayload
{
	unsigned long long PreHash;    ///< Client's own hash before any correction was applied.
	unsigned long long PostHash;   ///< After it, or equal to PreHash when none was sent.
	unsigned int Applied;
	unsigned int Pad;
};
#pragma pack(pop)

/**
 * @brief FNV-1a over raw bytes, for fingerprinting the scene both peers claim to have built.
 * @param h Accumulator, updated in place.
 * @param data First byte.
 * @param bytes Length.
 */
void HashBytes(unsigned long long& h, const void* data, size_t bytes)
{
	const unsigned char* p = (const unsigned char*)data;
	for (size_t i = 0; i < bytes; ++i)
	{
		h ^= (unsigned long long)p[i];
		h *= 1099511628211ULL;
	}
}

/**
 * @brief Fingerprint of everything a backend reads out of a SceneSpec.
 * @param s Scene.
 * @return Hash both peers must agree on before any tick is compared.
 */
unsigned long long SceneHash(const SceneSpec& s)
{
	unsigned long long h = 14695981039346656037ULL;
	if (!s.BoxPositions.empty())
		HashBytes(h, &s.BoxPositions[0], s.BoxPositions.size() * sizeof(float));
	if (!s.Constraints.empty())
		HashBytes(h, &s.Constraints[0], s.Constraints.size() * sizeof(DemoConstraint));
	HashBytes(h, &s.GroundHalfExtent, sizeof(s.GroundHalfExtent));
	HashBytes(h, &s.IslandCount, sizeof(s.IslandCount));
	HashBytes(h, s.WorldOffset, sizeof(s.WorldOffset));
	return h;
}

/// One reliable ordered link to exactly one peer, framed as {header, payload}.
class TcpLink
{
public:
	TcpLink() : m_socket(INVALID_SOCKET), m_listen(INVALID_SOCKET), m_started(false), m_cancel(false) { }
	~TcpLink() { Close(); }

	/**
	 * @brief Brings Winsock up.
	 * @return False if Winsock refused to start.
	 */
	bool Start()
	{
		WSADATA data;
		if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
		{
			printf("Net: WSAStartup failed (%d)\n", WSAGetLastError());
			return false;
		}
		m_cancel.store(false);
		m_started = true;
		return true;
	}

	/**
	 * @brief Drops the sockets to release a thread parked in Accept/Dial/Receive, leaving Winsock up.
	 *
	 * Separate from Close so a caller can join that thread BEFORE WSACleanup runs; tearing Winsock
	 * down under a live socket call is what made a disconnect-while-connecting hang.
	 */
	void Cancel()
	{
		m_cancel.store(true);
		if (m_socket != INVALID_SOCKET)
		{
			shutdown(m_socket, SD_BOTH);
			closesocket(m_socket);
		}
		if (m_listen != INVALID_SOCKET)
			closesocket(m_listen);
		m_socket = m_listen = INVALID_SOCKET;
	}

	/**
	 * @brief Listens on a port and waits for one peer.
	 * @param port TCP port.
	 * @param timeoutSeconds How long to wait for a connection.
	 * @return False on any socket error or timeout.
	 */
	bool Accept(int port, int timeoutSeconds)
	{
		m_listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (m_listen == INVALID_SOCKET)
		{
			printf("Net: socket() failed (%d)\n", WSAGetLastError());
			return false;
		}
		BOOL reuse = TRUE;
		setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));

		sockaddr_in addr;
		memset(&addr, 0, sizeof(addr));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = htonl(INADDR_ANY);
		addr.sin_port = htons((unsigned short)port);
		if (bind(m_listen, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
		{
			printf("Net: bind on port %d failed (%d)\n", port, WSAGetLastError());
			return false;
		}
		if (listen(m_listen, 1) == SOCKET_ERROR)
		{
			printf("Net: listen failed (%d)\n", WSAGetLastError());
			return false;
		}

		printf("Net: listening on port %d, waiting up to %ds for a peer...\n", port, timeoutSeconds);
		fflush(stdout);

		fd_set readable;
		FD_ZERO(&readable);
		FD_SET(m_listen, &readable);
		timeval wait;
		wait.tv_sec = timeoutSeconds;
		wait.tv_usec = 0;
		const int ready = select(0, &readable, 0, 0, &wait);
		if (ready <= 0)
		{
			printf("Net: no peer connected within %ds\n", timeoutSeconds);
			return false;
		}

		m_socket = accept(m_listen, 0, 0);
		closesocket(m_listen);
		m_listen = INVALID_SOCKET;
		if (m_socket == INVALID_SOCKET)
		{
			printf("Net: accept failed (%d)\n", WSAGetLastError());
			return false;
		}
		Configure();
		printf("Net: peer connected.\n");
		fflush(stdout);
		return true;
	}

	/**
	 * @brief Dials a listening peer, retrying until it comes up.
	 * @param host Address or name.
	 * @param port TCP port.
	 * @param timeoutSeconds How long to keep retrying.
	 * @return False if the peer never answered.
	 */
	bool Dial(const char* host, int port, int timeoutSeconds)
	{
		char service[16];
		sprintf(service, "%d", port);
		addrinfo hints;
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;
		addrinfo* resolved = 0;
		if (getaddrinfo(host, service, &hints, &resolved) != 0 || !resolved)
		{
			printf("Net: cannot resolve '%s'\n", host);
			return false;
		}

		printf("Net: connecting to %s:%d (up to %ds)...\n", host, port, timeoutSeconds);
		fflush(stdout);
		const DWORD deadline = GetTickCount() + (DWORD)timeoutSeconds * 1000u;
		for (;;)
		{
			if (m_cancel.load())
			{
				freeaddrinfo(resolved);
				return false;
			}
			m_socket = socket(resolved->ai_family, resolved->ai_socktype, resolved->ai_protocol);
			if (m_socket != INVALID_SOCKET
				&& connect(m_socket, resolved->ai_addr, (int)resolved->ai_addrlen) != SOCKET_ERROR)
				break;
			if (m_socket != INVALID_SOCKET)
			{
				closesocket(m_socket);
				m_socket = INVALID_SOCKET;
			}
			if (GetTickCount() >= deadline)
			{
				printf("Net: could not connect to %s:%d\n", host, port);
				freeaddrinfo(resolved);
				return false;
			}
			Sleep(250);
		}
		freeaddrinfo(resolved);
		Configure();
		printf("Net: connected.\n");
		fflush(stdout);
		return true;
	}

	/**
	 * @brief Writes one framed message.
	 * @param type One of NetMsg.
	 * @param tick Tick the message describes; repurposed as an index by MSG_BEGIN.
	 * @param data Payload, may be null.
	 * @param bytes Payload length.
	 * @return False if the link broke.
	 */
	bool Send(unsigned int type, unsigned long long tick, const void* data, size_t bytes)
	{
		FrameHeader header;
		header.Magic = kNetMagic;
		header.Type = type;
		header.Tick = tick;
		header.Length = (unsigned int)bytes;
		if (!SendAll((const unsigned char*)&header, sizeof(header)))
			return false;
		return bytes == 0 || SendAll((const unsigned char*)data, bytes);
	}

	/**
	 * @brief Reads one framed message, blocking until it arrives or the socket times out.
	 * @param type Receives the message type.
	 * @param tick Receives the tick field.
	 * @param payload Receives the payload; resized to the framed length.
	 * @return False on timeout, a broken link, or a bad frame.
	 */
	bool Receive(unsigned int& type, unsigned long long& tick, std::vector<unsigned char>& payload)
	{
		FrameHeader header;
		if (!RecvAll((unsigned char*)&header, sizeof(header)))
			return false;
		if (header.Magic != kNetMagic || header.Length > kMaxPayloadBytes)
		{
			printf("Net: bad frame (magic %08x, length %u)\n", header.Magic, header.Length);
			return false;
		}
		type = header.Type;
		tick = header.Tick;
		payload.resize(header.Length);
		return header.Length == 0 || RecvAll(&payload[0], header.Length);
	}

	void Close()
	{
		Cancel();
		if (m_started)
			WSACleanup();
		m_started = false;
	}

private:
	/// Nagle would add a coalescing delay to every barrier round trip, which are tiny by design.
	void Configure()
	{
		BOOL noDelay = TRUE;
		setsockopt(m_socket, IPPROTO_TCP, TCP_NODELAY, (const char*)&noDelay, sizeof(noDelay));
		DWORD timeout = kRecvTimeoutMs;
		setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
	}

	bool SendAll(const unsigned char* p, size_t n)
	{
		while (n)
		{
			const int sent = send(m_socket, (const char*)p, (int)(n > 65536 ? 65536 : n), 0);
			if (sent <= 0)
			{
				printf("Net: send failed (%d)\n", WSAGetLastError());
				return false;
			}
			p += sent;
			n -= sent;
		}
		return true;
	}

	bool RecvAll(unsigned char* p, size_t n)
	{
		while (n)
		{
			const int got = recv(m_socket, (char*)p, (int)(n > 65536 ? 65536 : n), 0);
			if (got == 0)
			{
				printf("Net: peer closed the link\n");
				return false;
			}
			if (got < 0)
			{
				printf("Net: recv failed (%d)%s\n", WSAGetLastError(),
					   WSAGetLastError() == WSAETIMEDOUT ? " - peer stopped responding" : "");
				return false;
			}
			p += got;
			n -= got;
		}
		return true;
	}

	SOCKET m_socket;
	SOCKET m_listen;
	bool m_started;
	std::atomic<bool> m_cancel;
};

/**
 * @brief Exchanges run settings and refuses a mismatch.
 *
 * A configuration difference would otherwise surface as a tick-0 divergence, which is the one
 * verdict this harness must never report for the wrong reason.
 *
 * @param link Connected link.
 * @param cfg Local settings.
 * @param scene Local scene.
 * @param steps Local step count.
 * @param backends Local backend list, compared by name.
 * @return False if the peers disagree about what they are running.
 */
bool Handshake(TcpLink& link, const DemoNetConfig& cfg, const SceneSpec& scene, int steps,
			   const std::vector<std::unique_ptr<PhysicsBackend> >& backends)
{
	std::string names;
	for (size_t i = 0; i < backends.size(); ++i)
	{
		names += backends[i]->Name();
		names += "\n";
	}

	std::vector<unsigned char> out(sizeof(HelloPayload) + names.size());
	HelloPayload mine;
	mine.Version = kNetVersion;
	mine.Mode = (unsigned int)cfg.Mode;
	mine.Interval = (unsigned int)cfg.Interval;
	mine.CorrectEvery = (unsigned int)cfg.CorrectEvery;
	mine.Steps = (unsigned int)steps;
	mine.BackendCount = (unsigned int)backends.size();
	mine.BodyCount = (unsigned int)scene.BoxCount();
	mine.SceneHash = SceneHash(scene);
	memcpy(&out[0], &mine, sizeof(mine));
	if (!names.empty())
		memcpy(&out[sizeof(mine)], names.c_str(), names.size());

	if (!link.Send(MSG_HELLO, 0, &out[0], out.size()))
		return false;

	unsigned int type = 0;
	unsigned long long tick = 0;
	std::vector<unsigned char> in;
	if (!link.Receive(type, tick, in))
		return false;
	if (type != MSG_HELLO || in.size() < sizeof(HelloPayload))
	{
		printf("Net: handshake: unexpected first message\n");
		return false;
	}

	HelloPayload theirs;
	memcpy(&theirs, &in[0], sizeof(theirs));
	const std::string theirNames((const char*)&in[0] + sizeof(theirs), in.size() - sizeof(theirs));

	bool ok = true;
	if (theirs.Version != mine.Version)
	{
		printf("Net: protocol version %u vs peer %u\n", mine.Version, theirs.Version);
		ok = false;
	}
	if (theirs.Mode != mine.Mode)
	{
		printf("Net: mode mismatch (--net-mode differs between the peers)\n");
		ok = false;
	}
	if (theirs.Interval != mine.Interval || theirs.CorrectEvery != mine.CorrectEvery)
	{
		printf("Net: interval/correction cadence mismatch (%u/%u vs %u/%u)\n", mine.Interval,
			   mine.CorrectEvery, theirs.Interval, theirs.CorrectEvery);
		ok = false;
	}
	if (theirs.Steps != mine.Steps)
	{
		printf("Net: --steps mismatch (%u vs %u)\n", mine.Steps, theirs.Steps);
		ok = false;
	}
	if (theirs.BodyCount != mine.BodyCount || theirs.SceneHash != mine.SceneHash)
	{
		printf("Net: scene mismatch - %u bodies/%016llx here, %u/%016llx there. Check --scene,\n"
			   "  --boxes and OSDEMO_WORLD_OFFSET on both peers.\n",
			   mine.BodyCount, mine.SceneHash, theirs.BodyCount, theirs.SceneHash);
		ok = false;
	}
	if (theirNames != names)
	{
		printf("Net: backend list mismatch - check --gpu-only on both peers.\n  here: %s  there: %s",
			   names.c_str(), theirNames.c_str());
		ok = false;
	}

	if (ok)
		printf("Net: handshake ok - %u bodies, scene %016llx, %u backends, %d steps, every %d ticks\n",
			   mine.BodyCount, mine.SceneHash, mine.BackendCount, steps, cfg.Interval);
	return ok;
}

/**
 * @brief Agrees with the peer on whether this backend is usable before any tick runs.
 * @param link Connected link.
 * @param index Backend index, echoed as the frame's tick so a lost step is caught.
 * @param resetOk Whether the local Reset succeeded.
 * @param canSnapshot Whether the local backend implements CaptureState.
 * @param peerCanSnapshot Receives the peer's snapshot capability.
 * @return False when either peer cannot run this backend, or the link broke.
 */
bool BeginBackend(TcpLink& link, size_t index, bool resetOk, bool canSnapshot,
				  bool& peerCanSnapshot)
{
	unsigned char mine[2] = {(unsigned char)(resetOk ? 1 : 0), (unsigned char)(canSnapshot ? 1 : 0)};
	if (!link.Send(MSG_BEGIN, index, mine, sizeof(mine)))
		return false;

	unsigned int type = 0;
	unsigned long long tick = 0;
	std::vector<unsigned char> in;
	if (!link.Receive(type, tick, in))
		return false;
	if (type != MSG_BEGIN || tick != index || in.size() < 2)
	{
		printf("Net: lost step with the peer at backend %u\n", (unsigned)index);
		return false;
	}
	peerCanSnapshot = in[1] != 0;
	return resetOk && in[0] != 0;
}

/**
 * @brief Bucket indices whose hashes differ between the peers.
 * @param mine Local bucket hashes.
 * @param theirs Peer bucket hashes.
 * @param out Receives ascending indices; cleared first.
 * @return False when the peers disagree on the bucket count, which makes any diff meaningless.
 */
bool DiffBuckets(const std::vector<unsigned long long>& mine,
				 const std::vector<unsigned long long>& theirs, std::vector<int>& out)
{
	out.clear();
	if (mine.empty() || mine.size() != theirs.size())
		return false;
	for (size_t i = 0; i < mine.size(); ++i)
		if (mine[i] != theirs[i])
			out.push_back((int)i);
	return true;
}

/**
 * @brief Reads a MSG_BUCKETS payload back into bucket hashes.
 * @param payload Wire bytes; a partial trailing element is dropped.
 * @param out Receives the hashes.
 */
void DecodeBuckets(const std::vector<unsigned char>& payload, std::vector<unsigned long long>& out)
{
	out.assign(payload.size() / sizeof(unsigned long long), 0);
	if (!out.empty())
		memcpy(&out[0], &payload[0], out.size() * sizeof(unsigned long long));
}

/**
 * @brief Local per-bucket hashes for the barrier just published.
 *
 * Lazy by design: the readback is only paid once a root mismatch is known, and the barrier holds the
 * sim at that tick so the backend's last BeginHashState still describes it.
 *
 * @param backend Backend to read.
 * @param out Receives the hashes; empty when this backend has no buckets.
 * @return False when bucketing is unsupported or produced nothing.
 */
bool LocalBuckets(PhysicsBackend* backend, std::vector<unsigned long long>& out)
{
	out.clear();
	if (backend->HashBucketCount() <= 0)
		return false;
	return backend->FetchHashBuckets(out) && !out.empty();
}

/**
 * @brief Packs bucket hashes for the wire.
 * @param buckets Hashes, possibly empty.
 * @param out Receives the bytes.
 */
void EncodeBuckets(const std::vector<unsigned long long>& buckets, std::vector<unsigned char>& out)
{
	out.resize(buckets.size() * sizeof(unsigned long long));
	if (!out.empty())
		memcpy(&out[0], &buckets[0], out.size());
}

/**
 * @brief Prints what healing cost and whether it worked.
 * @param stats Counters to report.
 * @param indent Leading whitespace, so this reads inside a per-backend block.
 */
void PrintHealStats(const DemoHealStats& stats, const char* indent)
{
	if (stats.Unsupported)
	{
		printf("%sHEALING UNAVAILABLE - the backend does not bucket its state hash.\n", indent);
		return;
	}
	printf("%s%d divergence(s), %d recovered; %d round(s), %d/%d buckets corrected, %llu bytes\n",
		   indent, stats.Divergences, stats.Recoveries, stats.Rounds, stats.BucketsCorrected,
		   stats.BucketsCompared, stats.BytesSent);
	if (stats.Recoveries)
		printf("%sticks to recovery: last %llu, worst %llu\n", indent, stats.LastTicksToRecovery,
			   stats.WorstTicksToRecovery);
	if (stats.DivergedSinceTick != 0)
		printf("%s*** STILL DIVERGED since tick %llu\n", indent, stats.DivergedSinceTick);
}

/**
 * @brief One bucket-narrowed correction, at a barrier both peers are held at.
 *
 * Ordered host-first rather than the root hash's symmetric write-then-read: a bucket vector can
 * exceed a socket buffer, where a symmetric exchange would deadlock.
 *
 * @param link Connected link.
 * @param backend Backend to narrow and correct.
 * @param tick Barrier tick both peers are held at.
 * @param isHost True on the authority, which captures rather than applies.
 * @param dtSeconds Seconds since the previous heal application, for the convergence solve.
 * @param stats Updated with round/bucket/byte counts.
 * @return False only on a link failure; an unsupported backend is reported and returns true.
 */
bool HealRound(TcpLink& link, PhysicsBackend* backend, unsigned long long tick, bool isHost,
			   double dtSeconds, DemoHealStats& stats)
{
	std::vector<unsigned long long> mine;
	LocalBuckets(backend, mine);
	std::vector<unsigned char> minePayload;
	EncodeBuckets(mine, minePayload);
	const void* mineFirst = minePayload.empty() ? 0 : &minePayload[0];

	unsigned int type = 0;
	unsigned long long gotTick = 0;
	std::vector<unsigned char> in;
	if (isHost)
	{
		if (!link.Send(MSG_BUCKETS, tick, mineFirst, minePayload.size())
			|| !link.Receive(type, gotTick, in))
			return false;
	}
	else
	{
		if (!link.Receive(type, gotTick, in)
			|| !link.Send(MSG_BUCKETS, tick, mineFirst, minePayload.size()))
			return false;
	}
	stats.BytesSent += minePayload.size();
	if (type != MSG_BUCKETS || gotTick != tick)
	{
		printf("Net: lost step exchanging buckets at tick %llu\n", tick);
		return false;
	}

	std::vector<unsigned long long> theirs;
	DecodeBuckets(in, theirs);
	std::vector<int> diverged;
	// Both peers diff the same pair of vectors, so they agree on whether healing can run at all -
	// which is what lets the MSG_HEAL below be skipped on both sides without desynchronising them.
	if (!DiffBuckets(mine, theirs, diverged))
	{
		if (!stats.Unsupported)
		{
			stats.Unsupported = true;
			printf("Net: healing unavailable - %s buckets its state hash (%u vs peer %u buckets).\n",
				   mine.empty() ? "neither peer" : "only one peer", (unsigned)mine.size(),
				   (unsigned)theirs.size());
			fflush(stdout);
		}
		return true;
	}

	++stats.Rounds;
	stats.BucketsCompared += (int)mine.size();
	stats.BucketsCorrected += (int)diverged.size();

	if (isHost)
	{
		std::vector<unsigned char> state;
		if (!diverged.empty() && !backend->CaptureBuckets(diverged, state))
			state.clear();
		if (!link.Send(MSG_HEAL, tick, state.empty() ? 0 : &state[0], state.size()))
			return false;
		stats.BytesSent += state.size();
		return true;
	}

	if (!link.Receive(type, gotTick, in))
		return false;
	if (type != MSG_HEAL || gotTick != tick)
	{
		printf("Net: lost step awaiting a correction at tick %llu\n", tick);
		return false;
	}
	if (!in.empty())
	{
		PhysicsBackend::HealSettings settings;
		backend->ApplyBucketsConverging(&in[0], in.size(), settings, dtSeconds);
	}
	return true;
}

/**
 * @brief Symmetric hash comparison, optionally narrowing and correcting a divergence.
 * @param link Connected link.
 * @param backends Backends to run in order.
 * @param scene Scene to run.
 * @param steps Steps per backend.
 * @param fixedStep Timestep.
 * @param interval Ticks between exchanges.
 * @param heal Whether a mismatch opens a bucket-narrowed correction instead of only being reported.
 * @param isHost True on the authority.
 * @return 0 when every backend ended in sync, 1 on a divergence outstanding at the end, 2 on a
 *         link failure.
 */
int RunLockstep(TcpLink& link, std::vector<std::unique_ptr<PhysicsBackend> >& backends,
				const SceneSpec& scene, int steps, float fixedStep, int interval, bool heal,
				bool isHost)
{
	int worst = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		const bool resetOk = backends[b]->Reset(scene);
		bool peerCanSnapshot = false;
		if (!BeginBackend(link, b, resetOk, false, peerCanSnapshot))
		{
			printf("  %-30s SKIPPED - reset failed on one of the peers\n", backends[b]->Name());
			worst = 1;
			continue;
		}

		long long firstDivergentTick = -1;
		unsigned long long mineAtDivergence = 0, theirsAtDivergence = 0;
		int samples = 0, matched = 0;
		DemoHealStats stats;
		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(fixedStep);
			if (((i + 1) % interval) != 0)
				continue;

			unsigned long long hash = 0;
			// Begin/Fetch rather than HashState: FetchHashBuckets reads whatever the last
			// BeginHashState computed, and the default Fetch is HashState anyway.
			backends[b]->BeginHashState();
			if (!backends[b]->FetchHashState(hash))
				continue;

			// Both peers write before either reads: 8 bytes fits any socket buffer, so this cannot
			// deadlock, and it makes the exchange one round trip instead of two.
			const unsigned long long tick = (unsigned long long)(i + 1);
			if (!link.Send(MSG_HASH, tick, &hash, sizeof(hash)))
				return 2;
			unsigned int type = 0;
			unsigned long long gotTick = 0;
			std::vector<unsigned char> in;
			if (!link.Receive(type, gotTick, in))
				return 2;
			if (type != MSG_HASH || gotTick != tick || in.size() != sizeof(hash))
			{
				printf("Net: lost step at tick %d\n", i + 1);
				return 2;
			}

			unsigned long long peerHash = 0;
			memcpy(&peerHash, &in[0], sizeof(peerHash));
			++samples;
			if (peerHash == hash)
			{
				++matched;
				if (stats.DivergedSinceTick != 0)
				{
					const unsigned long long took = tick - stats.DivergedSinceTick;
					stats.LastTicksToRecovery = took;
					if (took > stats.WorstTicksToRecovery)
						stats.WorstTicksToRecovery = took;
					++stats.Recoveries;
					stats.DivergedSinceTick = 0;
					printf("    RECOVERED at tick %llu (%llu ticks after divergence)\n", tick, took);
					fflush(stdout);
				}
				continue;
			}

			if (firstDivergentTick < 0)
			{
				firstDivergentTick = (long long)tick;
				mineAtDivergence = hash;
				theirsAtDivergence = peerHash;
			}
			if (!heal)
				continue;
			if (stats.DivergedSinceTick == 0)
			{
				stats.DivergedSinceTick = tick;
				++stats.Divergences;
			}
			if (!HealRound(link, backends[b].get(), tick, isHost, (double)interval * fixedStep,
						   stats))
				return 2;
		}

		printf("  %-30s %d samples, %d matched", backends[b]->Name(), samples, matched);
		if (firstDivergentTick >= 0)
		{
			printf("\n    *** DESYNC: first divergent tick %lld  local %016llx  peer %016llx\n",
				   firstDivergentTick, mineAtDivergence, theirsAtDivergence);
			if (heal)
				PrintHealStats(stats, "    ");
			if (stats.DivergedSinceTick != 0 || !heal)
				worst = 1;
		}
		else if (samples > 0)
			printf("  IN SYNC\n");
		else
			printf("  NO SAMPLES - backend produced no hash\n");
		if (heal && firstDivergentTick < 0 && stats.Rounds != 0)
		{
			printf("    *** %d heal rounds ran without any divergence - healing is not free.\n",
				   stats.Rounds);
			worst = 1;
		}
		fflush(stdout);
	}

	if (!heal)
		printf("\n%s\n", worst ? "LOCKSTEP VERDICT: DESYNC DETECTED"
							   : "LOCKSTEP VERDICT: peers stayed in sync.");
	else
		printf("\n%s\n", worst ? "HEALING VERDICT: a divergence was still outstanding at the end."
							   : "HEALING VERDICT: peers ended in sync.");
	return worst;
}

/**
 * @brief Host side of the authoritative policy: snapshot, send, and score the client's replies.
 * @param link Connected link.
 * @param backends Backends to run in order.
 * @param scene Scene to run.
 * @param steps Steps per backend.
 * @param fixedStep Timestep.
 * @param interval Ticks between exchanges.
 * @param correctEvery Exchanges between snapshots.
 * @return 0 when the client tracked the host, 1 on drift or a correction that did not land,
 *         2 on a link failure or a backend that cannot snapshot.
 */
int RunAuthoritativeHost(TcpLink& link, std::vector<std::unique_ptr<PhysicsBackend> >& backends,
						 const SceneSpec& scene, int steps, float fixedStep, int interval,
						 int correctEvery)
{
	int worst = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		const bool resetOk = backends[b]->Reset(scene);
		std::vector<unsigned char> probe;
		const bool canSnapshot = resetOk && backends[b]->CaptureState(probe);
		bool peerCanSnapshot = false;
		if (!BeginBackend(link, b, resetOk, canSnapshot, peerCanSnapshot))
		{
			printf("  %-30s SKIPPED - reset failed on one of the peers\n", backends[b]->Name());
			worst = worst > 2 ? worst : 2;
			continue;
		}
		if (!canSnapshot || !peerCanSnapshot)
		{
			printf("  %-30s AUTHORITATIVE MODE UNAVAILABLE - CaptureState/ApplyState not\n"
				   "    implemented by this backend%s\n", backends[b]->Name(),
				   canSnapshot ? " on the client" : "");
			worst = worst > 2 ? worst : 2;
			continue;
		}

		int samples = 0, matched = 0, corrections = 0, landed = 0, inert = 0;
		int driftRun = 0, worstDriftRun = 0;
		long long firstDriftTick = -1;
		size_t snapshotBytes = 0;
		int sampleIndex = 0;
		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(fixedStep);
			if (((i + 1) % interval) != 0)
				continue;

			unsigned long long hash = 0;
			if (!backends[b]->HashState(hash))
				continue;
			++sampleIndex;

			std::vector<unsigned char> state;
			const bool correct = (sampleIndex % correctEvery) == 0
								 && backends[b]->CaptureState(state);
			std::vector<unsigned char> payload(sizeof(hash) + (correct ? state.size() : 0));
			memcpy(&payload[0], &hash, sizeof(hash));
			if (correct && !state.empty())
			{
				memcpy(&payload[sizeof(hash)], &state[0], state.size());
				snapshotBytes = state.size();
			}

			if (!link.Send(MSG_TICK, (unsigned long long)(i + 1), &payload[0], payload.size()))
				return 2;

			unsigned int type = 0;
			unsigned long long tick = 0;
			std::vector<unsigned char> in;
			if (!link.Receive(type, tick, in))
				return 2;
			if (type != MSG_ACK || tick != (unsigned long long)(i + 1)
				|| in.size() != sizeof(AckPayload))
			{
				printf("Net: lost step at tick %d\n", i + 1);
				return 2;
			}
			AckPayload ack;
			memcpy(&ack, &in[0], sizeof(ack));

			++samples;
			if (ack.PreHash == hash)
			{
				++matched;
				driftRun = 0;
			}
			else
			{
				++driftRun;
				if (driftRun > worstDriftRun)
					worstDriftRun = driftRun;
				if (firstDriftTick < 0)
					firstDriftTick = i + 1;
			}
			if (correct)
			{
				++corrections;
				if (ack.Applied && ack.PostHash == hash)
				{
					++landed;
					driftRun = 0;
				}
				else if (ack.Applied && ack.PostHash == ack.PreHash)
					++inert;
			}
		}

		if (!link.Send(MSG_END, (unsigned long long)steps, 0, 0))
			return 2;

		printf("  %-30s %d samples, %d agreed before correction, %d/%d corrections landed\n",
			   backends[b]->Name(), samples, matched, landed, corrections);
		printf("    snapshot %u bytes, longest drift run %d samples (%d ticks)",
			   (unsigned)snapshotBytes, worstDriftRun, worstDriftRun * interval);
		if (firstDriftTick >= 0)
			printf(", first drift at tick %lld", firstDriftTick);
		printf("\n");
		if (matched != samples)
			worst = worst > 1 ? worst : 1;
		if (landed != corrections)
		{
			printf("    *** a correction did NOT land: the client's state after ApplyState still\n"
				   "        hashes differently from the host's at the same tick.\n");
			// Separates a snapshot that restored the wrong state from one that reached nothing
			// HashState reads at all - a different bug in a different place.
			if (inert)
				printf("    *** %d of them left the client's hash BIT-IDENTICAL to before the\n"
					   "        apply: ApplyState returned true but changed nothing HashState sees.\n",
					   inert);
			worst = worst > 1 ? worst : 1;
		}
		fflush(stdout);
	}

	printf("\n%s\n", worst == 0 ? "AUTHORITATIVE VERDICT (host): client tracked the host exactly."
				   : worst == 1 ? "AUTHORITATIVE VERDICT (host): DRIFT DETECTED between corrections."
								: "AUTHORITATIVE VERDICT (host): could not run - see above.");
	return worst;
}

/**
 * @brief Client side: steps to the host's tick, scores its own state, then takes the correction.
 *
 * Corrections are HARD applies. PhysicsBackend exposes only opaque CaptureState/ApplyState bytes,
 * so the demo cannot steer toward the authority the way the engine's HealSettings does.
 *
 * @param link Connected link.
 * @param backends Backends to run in order.
 * @param scene Scene to run.
 * @param steps Steps per backend.
 * @param fixedStep Timestep.
 * @return 0 when this client matched the host, 1 on drift, 2 on a link or capability failure.
 */
int RunAuthoritativeClient(TcpLink& link, std::vector<std::unique_ptr<PhysicsBackend> >& backends,
						   const SceneSpec& scene, int steps, float fixedStep)
{
	int worst = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		const bool resetOk = backends[b]->Reset(scene);
		std::vector<unsigned char> probe;
		const bool canSnapshot = resetOk && backends[b]->CaptureState(probe);
		bool peerCanSnapshot = false;
		if (!BeginBackend(link, b, resetOk, canSnapshot, peerCanSnapshot))
		{
			printf("  %-30s SKIPPED - reset failed on one of the peers\n", backends[b]->Name());
			worst = worst > 2 ? worst : 2;
			continue;
		}
		if (!canSnapshot || !peerCanSnapshot)
		{
			printf("  %-30s AUTHORITATIVE MODE UNAVAILABLE - CaptureState/ApplyState not\n"
				   "    implemented by this backend%s\n", backends[b]->Name(),
				   canSnapshot ? " on the host" : "");
			worst = worst > 2 ? worst : 2;
			continue;
		}

		long long localTick = 0;
		int samples = 0, matched = 0, corrections = 0, landed = 0, failedApply = 0, inert = 0;
		int driftRun = 0, worstDriftRun = 0;
		long long firstDriftTick = -1;
		for (;;)
		{
			unsigned int type = 0;
			unsigned long long tick = 0;
			std::vector<unsigned char> in;
			if (!link.Receive(type, tick, in))
				return 2;
			if (type == MSG_END)
				break;
			if (type != MSG_TICK || in.size() < sizeof(unsigned long long))
			{
				printf("Net: lost step (unexpected message %u)\n", type);
				return 2;
			}

			while (localTick < (long long)tick && localTick < steps)
			{
				backends[b]->Step(fixedStep);
				++localTick;
			}

			unsigned long long hostHash = 0;
			memcpy(&hostHash, &in[0], sizeof(hostHash));

			AckPayload ack;
			ack.PreHash = 0;
			ack.Applied = 0;
			ack.Pad = 0;
			backends[b]->HashState(ack.PreHash);
			ack.PostHash = ack.PreHash;

			++samples;
			if (ack.PreHash == hostHash)
			{
				++matched;
				driftRun = 0;
			}
			else
			{
				++driftRun;
				if (driftRun > worstDriftRun)
					worstDriftRun = driftRun;
				if (firstDriftTick < 0)
					firstDriftTick = (long long)tick;
			}

			if (in.size() > sizeof(hostHash))
			{
				++corrections;
				if (backends[b]->ApplyState(&in[sizeof(hostHash)], in.size() - sizeof(hostHash)))
				{
					ack.Applied = 1;
					backends[b]->HashState(ack.PostHash);
					if (ack.PostHash == hostHash)
					{
						++landed;
						driftRun = 0;
					}
					else if (ack.PostHash == ack.PreHash)
						++inert;
				}
				else
					++failedApply;
			}

			if (!link.Send(MSG_ACK, tick, &ack, sizeof(ack)))
				return 2;
		}

		printf("  %-30s %d samples, %d agreed before correction, %d/%d corrections landed\n",
			   backends[b]->Name(), samples, matched, landed, corrections);
		printf("    longest drift run %d samples", worstDriftRun);
		if (firstDriftTick >= 0)
			printf(", first drift at tick %lld", firstDriftTick);
		if (failedApply)
			printf(", %d ApplyState calls REFUSED the payload", failedApply);
		if (inert)
			printf(", %d left the hash BIT-IDENTICAL to before the apply", inert);
		printf("\n");
		if (matched != samples)
			worst = worst > 1 ? worst : 1;
		if (landed != corrections)
			worst = worst > 1 ? worst : 1;
		fflush(stdout);
	}

	printf("\n%s\n", worst == 0 ? "AUTHORITATIVE VERDICT (client): matched the host exactly."
				   : worst == 1 ? "AUTHORITATIVE VERDICT (client): DRIFT DETECTED between corrections."
								: "AUTHORITATIVE VERDICT (client): could not run - see above.");
	return worst;
}
}  // namespace

int RunNetSync(std::vector<std::unique_ptr<PhysicsBackend> >& backends, const SceneSpec& scene,
			   int steps, float fixedStep, const DemoNetConfig& cfg)
{
	if (backends.empty())
	{
		printf("Net: no backends available.\n");
		return 2;
	}
	int interval = cfg.Interval < 1 ? 1 : cfg.Interval;
	const int correctEvery = cfg.CorrectEvery < 1 ? 1 : cfg.CorrectEvery;

	TcpLink link;
	if (!link.Start())
		return 2;
	if (cfg.Host ? !link.Accept(cfg.Port, cfg.ConnectTimeoutSeconds)
				 : !link.Dial(cfg.PeerHost.c_str(), cfg.Port, cfg.ConnectTimeoutSeconds))
		return 2;

	DemoNetConfig effective = cfg;
	effective.Interval = interval;
	effective.CorrectEvery = correctEvery;
	if (!Handshake(link, effective, scene, steps, backends))
	{
		printf("\nNET VERDICT: handshake refused - the peers are not running the same test.\n");
		return 2;
	}

	printf("\nNet %s run as %s: %s (%d bodies), %d steps, exchange every %d ticks\n",
		   cfg.Mode == DEMO_NET_LOCKSTEP ? "lockstep"
		   : cfg.Mode == DEMO_NET_HEALING ? "healing" : "authoritative",
		   cfg.Host ? "host" : "client", scene.Name, scene.BoxCount(), steps, interval);
	if (cfg.Mode == DEMO_NET_AUTHORITATIVE)
		printf("Corrections are HARD state applies every %d exchanges; the demo has no transform-level\n"
			   "  API to converge toward the authority the way the engine's HealSettings does.\n",
			   correctEvery);
	if (cfg.Mode == DEMO_NET_HEALING)
		printf("Nothing but an 8-byte root hash crosses the wire while in sync; a mismatch then costs\n"
			   "  one bucket-hash exchange plus only the diverged buckets' bodies.\n");
	fflush(stdout);

	if (cfg.Mode == DEMO_NET_LOCKSTEP || cfg.Mode == DEMO_NET_HEALING)
		return RunLockstep(link, backends, scene, steps, fixedStep, interval,
						   cfg.Mode == DEMO_NET_HEALING, cfg.Host);
	if (cfg.Host)
		return RunAuthoritativeHost(link, backends, scene, steps, fixedStep, interval, correctEvery);
	return RunAuthoritativeClient(link, backends, scene, steps, fixedStep);
}

// ---------------------------------------------------------------------------------------------
// Interactive session. Only the worker thread ever calls Receive and only the render thread ever
// calls Send - TCP is full duplex, so that needs no lock of its own.
// ---------------------------------------------------------------------------------------------

enum NetLinkState
{
	NET_CONNECTING = 0,
	NET_CONNECTED,
	NET_FAILED
};

/// Where a heal round has got to. Every phase but IDLE holds stepping at the barrier.
enum HealPhase
{
	HEAL_IDLE = 0,
	HEAL_SEND_BUCKETS,   ///< Render thread owes the peer its bucket hashes.
	HEAL_WAIT_BUCKETS,   ///< Sent; waiting for the peer's, which the worker parks.
	HEAL_CLIENT_WAIT     ///< Client only: buckets diffed, waiting for the authority's payload.
};

struct DemoNetInteractiveSession::Impl
{
	Impl() : Interval(50), StepSeconds(1.0 / 60.0), Heal(false), IsHost(false), Running(false),
			 State(NET_CONNECTING), Desynced(false), FirstDivergent(0), Matched(0), PeerTick(0),
			 Phase(HEAL_IDLE), HealingBodies(0), HealTick(0), PeerBucketTick(0),
			 PeerBucketsReady(false), HealPayloadTick(0), HealPayloadReady(false) { Status[0] = 0; }

	TcpLink Link;
	int Interval;
	double StepSeconds;
	bool Heal;
	bool IsHost;
	std::thread Worker;
	std::atomic<bool> Running;
	std::atomic<int> State;
	/// Fixed buffer, written only by the worker before a State transition the render thread reads
	/// after - so the HUD never sees a torn string.
	char Status[96];
	std::atomic<bool> Desynced;
	std::atomic<unsigned long long> FirstDivergent;
	std::atomic<int> Matched;
	std::atomic<unsigned long long> PeerTick;
	std::atomic<int> Phase;
	std::atomic<int> HealingBodies;

	/// Render thread only: this barrier's local bucket hashes, kept for the diff a frame later.
	std::vector<unsigned long long> MyBuckets;

	mutable std::mutex Mutex;
	std::map<unsigned long long, unsigned long long> PeerHashes;   ///< barrier tick -> peer hash
	std::map<unsigned long long, unsigned long long> LocalHashes;
	DemoHealStats Stats;
	unsigned long long HealTick;
	std::vector<unsigned long long> PeerBuckets;
	unsigned long long PeerBucketTick;
	bool PeerBucketsReady;
	std::vector<unsigned char> HealPayload;
	unsigned long long HealPayloadTick;
	bool HealPayloadReady;

	/**
	 * @brief Compares a barrier tick once both sides' hashes are in, and opens a heal round.
	 *
	 * Called under Mutex from whichever side arrives second, so the comparison happens exactly once
	 * regardless of which peer was ahead.
	 *
	 * @param tick Barrier tick to score.
	 */
	void CompareLocked(unsigned long long tick)
	{
		std::map<unsigned long long, unsigned long long>::const_iterator local = LocalHashes.find(tick);
		std::map<unsigned long long, unsigned long long>::const_iterator peer = PeerHashes.find(tick);
		if (local == LocalHashes.end() || peer == PeerHashes.end())
			return;
		if (local->second == peer->second)
		{
			Matched.store(Matched.load() + 1);
			if (Stats.DivergedSinceTick != 0)
			{
				const unsigned long long took = tick - Stats.DivergedSinceTick;
				Stats.LastTicksToRecovery = took;
				if (took > Stats.WorstTicksToRecovery)
					Stats.WorstTicksToRecovery = took;
				++Stats.Recoveries;
				Stats.DivergedSinceTick = 0;
				printf("RECOVERED at tick %llu (%llu ticks after divergence)\n", tick, took);
				fflush(stdout);
			}
			return;
		}
		if (!Desynced.exchange(true))
		{
			FirstDivergent.store(tick);
			printf("\nDESYNC at tick %llu: local %016llx peer %016llx\n", tick, local->second,
				   peer->second);
			fflush(stdout);
		}
		// Tracked even with healing off, so "still diverged at the end" means the same thing in both
		// modes - which is what the exit code keys off.
		if (Stats.DivergedSinceTick == 0)
		{
			Stats.DivergedSinceTick = tick;
			++Stats.Divergences;
		}
		if (!Heal || Stats.Unsupported)
			return;
		HealTick = tick;
		Phase.store(HEAL_SEND_BUCKETS);
	}
};

DemoNetInteractiveSession::DemoNetInteractiveSession() : m_impl(new Impl())
{
}

DemoNetInteractiveSession::~DemoNetInteractiveSession()
{
	m_impl->Running.store(false);
	m_impl->Link.Cancel();   // unblocks the worker's accept/dial/recv, Winsock still up
	if (m_impl->Worker.joinable())
		m_impl->Worker.join();
	m_impl->Link.Close();
	delete m_impl;
}

DemoNetInteractiveSession* DemoNetInteractiveSession::Begin(
	const DemoNetConfig& cfg, const SceneSpec& scene,
	const std::vector<std::unique_ptr<PhysicsBackend> >& backends)
{
	DemoNetInteractiveSession* session = new DemoNetInteractiveSession();
	Impl* impl = session->m_impl;
	impl->Interval = cfg.Interval > 0 ? cfg.Interval : 50;
	impl->StepSeconds = cfg.StepSeconds > 0.0 ? cfg.StepSeconds : 1.0 / 60.0;
	impl->Heal = cfg.Mode == DEMO_NET_HEALING;
	impl->IsHost = cfg.Host;
	impl->Running.store(true);
	_snprintf_s(impl->Status, sizeof(impl->Status), _TRUNCATE, "%s (sim held at tick 0)",
				cfg.Host ? "listening for a peer..." : "connecting...");

	// Steps is 0: an interactive run has no agreed length, so the handshake only has to agree on
	// scene, cadence and backends.
	DemoNetConfig local = cfg;
	if (local.Mode != DEMO_NET_HEALING)
		local.Mode = DEMO_NET_LOCKSTEP;
	const SceneSpec sceneCopy = scene;
	const std::vector<std::unique_ptr<PhysicsBackend> >* backendList = &backends;

	impl->Worker = std::thread([impl, local, sceneCopy, backendList]() {
		if (!impl->Link.Start()
			|| !(local.Host ? impl->Link.Accept(local.Port, local.ConnectTimeoutSeconds)
							: impl->Link.Dial(local.PeerHost.c_str(), local.Port,
											  local.ConnectTimeoutSeconds))
			|| !Handshake(impl->Link, local, sceneCopy, 0, *backendList))
		{
			_snprintf_s(impl->Status, sizeof(impl->Status), _TRUNCATE,
						"%s", local.Host ? "no peer connected" : "could not connect");
			impl->State.store(NET_FAILED);
			return;
		}

		_snprintf_s(impl->Status, sizeof(impl->Status), _TRUNCATE, "%s, barrier every %d ticks%s",
					local.Host ? "host" : "joined", impl->Interval, impl->Heal ? ", healing" : "");
		impl->State.store(NET_CONNECTED);

		while (impl->Running.load())
		{
			unsigned int type = 0;
			unsigned long long tick = 0;
			std::vector<unsigned char> payload;
			if (!impl->Link.Receive(type, tick, payload))
			{
				if (impl->Running.load())
					_snprintf_s(impl->Status, sizeof(impl->Status), _TRUNCATE, "link lost");
				impl->State.store(NET_FAILED);
				break;
			}
			// Parked, not acted on: every heal phase calls the backend, which belongs to the render
			// thread, so this thread only ever delivers.
			if (type == MSG_BUCKETS)
			{
				std::lock_guard<std::mutex> lock(impl->Mutex);
				DecodeBuckets(payload, impl->PeerBuckets);
				impl->PeerBucketTick = tick;
				impl->PeerBucketsReady = true;
				continue;
			}
			if (type == MSG_HEAL)
			{
				std::lock_guard<std::mutex> lock(impl->Mutex);
				impl->HealPayload = payload;
				impl->HealPayloadTick = tick;
				impl->HealPayloadReady = true;
				continue;
			}
			if (type != MSG_HASH || payload.size() < sizeof(unsigned long long))
				continue;

			unsigned long long hash = 0;
			memcpy(&hash, &payload[0], sizeof(hash));
			impl->PeerTick.store(tick);
			std::lock_guard<std::mutex> lock(impl->Mutex);
			impl->PeerHashes[tick] = hash;
			impl->CompareLocked(tick);
		}
	});

	return session;
}

bool DemoNetInteractiveSession::MayStep(unsigned long long tick) const
{
	// Held at tick 0 until the peer is actually there: stepping while listening would put this side
	// thousands of ticks ahead, and the barrier could then never match. After a drop we run free.
	const int state = m_impl->State.load();
	if (state == NET_CONNECTING)
		return false;
	if (state != NET_CONNECTED)
		return true;
	if (tick == 0 || (tick % (unsigned long long)m_impl->Interval) != 0)
		return true;
	// Held for the whole heal handshake as well as the hash: the authority must still BE at this
	// tick when it captures, or the payload would describe a state its own hash never named.
	if (m_impl->Phase.load() != HEAL_IDLE)
		return false;
	std::lock_guard<std::mutex> lock(m_impl->Mutex);
	return m_impl->PeerHashes.find(tick) != m_impl->PeerHashes.end();
}

void DemoNetInteractiveSession::PumpHeal(PhysicsBackend* backend)
{
	Impl* impl = m_impl;
	if (!backend || impl->State.load() != NET_CONNECTED || impl->Phase.load() == HEAL_IDLE)
		return;

	if (impl->Phase.load() == HEAL_SEND_BUCKETS)
	{
		unsigned long long tick = 0;
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			tick = impl->HealTick;
		}
		LocalBuckets(backend, impl->MyBuckets);
		std::vector<unsigned char> payload;
		EncodeBuckets(impl->MyBuckets, payload);
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			impl->Stats.BytesSent += payload.size();
		}
		// A send failure needs no handling here: the worker's own recv sees the same broken link and
		// moves the session to NET_FAILED, which ungates stepping.
		impl->Link.Send(MSG_BUCKETS, tick, payload.empty() ? 0 : &payload[0], payload.size());
		impl->Phase.store(HEAL_WAIT_BUCKETS);
	}

	if (impl->Phase.load() == HEAL_WAIT_BUCKETS)
	{
		std::vector<unsigned long long> theirs;
		unsigned long long tick = 0;
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			if (!impl->PeerBucketsReady || impl->PeerBucketTick != impl->HealTick)
				return;
			theirs = impl->PeerBuckets;
			tick = impl->HealTick;
			impl->PeerBucketsReady = false;
		}

		std::vector<int> diverged;
		// Both peers diff the same pair of vectors, so they agree on whether healing can run - which
		// is what lets the payload below be skipped on both sides without desynchronising them.
		if (!DiffBuckets(impl->MyBuckets, theirs, diverged))
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			if (!impl->Stats.Unsupported)
			{
				impl->Stats.Unsupported = true;
				printf("Net: healing unavailable - %s buckets its state hash (%u vs peer %u).\n",
					   impl->MyBuckets.empty() ? "neither peer" : "only one peer",
					   (unsigned)impl->MyBuckets.size(), (unsigned)theirs.size());
				fflush(stdout);
			}
			impl->Phase.store(HEAL_IDLE);
			return;
		}
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			++impl->Stats.Rounds;
			impl->Stats.BucketsCompared += (int)impl->MyBuckets.size();
			impl->Stats.BucketsCorrected += (int)diverged.size();
		}

		if (impl->IsHost)
		{
			std::vector<unsigned char> state;
			if (!diverged.empty() && !backend->CaptureBuckets(diverged, state))
				state.clear();
			{
				std::lock_guard<std::mutex> lock(impl->Mutex);
				impl->Stats.BytesSent += state.size();
			}
			impl->Link.Send(MSG_HEAL, tick, state.empty() ? 0 : &state[0], state.size());
			impl->Phase.store(HEAL_IDLE);
			return;
		}
		impl->Phase.store(HEAL_CLIENT_WAIT);
	}

	if (impl->Phase.load() == HEAL_CLIENT_WAIT)
	{
		std::vector<unsigned char> payload;
		{
			std::lock_guard<std::mutex> lock(impl->Mutex);
			if (!impl->HealPayloadReady || impl->HealPayloadTick != impl->HealTick)
				return;
			payload.swap(impl->HealPayload);
			impl->HealPayloadReady = false;
		}
		if (!payload.empty())
		{
			PhysicsBackend::HealSettings settings;
			backend->ApplyBucketsConverging(&payload[0], payload.size(), settings,
											impl->StepSeconds * (double)impl->Interval);
		}
		impl->HealingBodies.store(backend->HealingBodyCount());
		impl->Phase.store(HEAL_IDLE);
	}
}

void DemoNetInteractiveSession::PublishHash(unsigned long long tick, unsigned long long hash)
{
	{
		std::lock_guard<std::mutex> lock(m_impl->Mutex);
		if (m_impl->LocalHashes.find(tick) != m_impl->LocalHashes.end())
			return;
		m_impl->LocalHashes[tick] = hash;
		m_impl->CompareLocked(tick);
	}
	if (m_impl->State.load() == NET_CONNECTED)
		m_impl->Link.Send(MSG_HASH, tick, &hash, sizeof(hash));
}

int DemoNetInteractiveSession::Interval() const { return m_impl->Interval; }
bool DemoNetInteractiveSession::Connecting() const { return m_impl->State.load() == NET_CONNECTING; }
bool DemoNetInteractiveSession::Connected() const { return m_impl->State.load() == NET_CONNECTED; }
bool DemoNetInteractiveSession::Failed() const { return m_impl->State.load() == NET_FAILED; }
const char* DemoNetInteractiveSession::StatusText() const { return m_impl->Status; }
bool DemoNetInteractiveSession::Desynced() const { return m_impl->Desynced.load(); }
unsigned long long DemoNetInteractiveSession::FirstDivergentTick() const { return m_impl->FirstDivergent.load(); }
int DemoNetInteractiveSession::Matched() const { return m_impl->Matched.load(); }
unsigned long long DemoNetInteractiveSession::PeerTick() const { return m_impl->PeerTick.load(); }
bool DemoNetInteractiveSession::HealingNow() const { return m_impl->Phase.load() != HEAL_IDLE; }
int DemoNetInteractiveSession::HealingBodies() const { return m_impl->HealingBodies.load(); }

DemoHealStats DemoNetInteractiveSession::HealStats() const
{
	std::lock_guard<std::mutex> lock(m_impl->Mutex);
	return m_impl->Stats;
}

bool DemoNetInteractiveSession::Diverged() const
{
	std::lock_guard<std::mutex> lock(m_impl->Mutex);
	return m_impl->Stats.DivergedSinceTick != 0;
}
