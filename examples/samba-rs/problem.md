# Problem: A Samba-equivalent network file/print server suite (Rust)

Build a Rust workspace that reimplements a substantial subset of the
[Samba](https://en.wikipedia.org/wiki/Samba_(software)) server suite — SMB/CIFS
file serving, NetBIOS name resolution, the DCE/RPC layers above SMB,
NTLM/Kerberos authentication, an LDAP/KDC/DNS stack for Active Directory
Domain Controller mode, and a small set of administrative tools. Treat the
Wikipedia *Features* section as the authoritative scope statement.

The wire protocols, on-disk formats, and RPC interfaces are public and
documented; you should mirror their structure faithfully. Authoritative
references:

- **MS-SMB2** — SMB 2.x / 3.x file protocol
- **MS-CIFS / MS-SMB** — legacy SMB1 (optional, off by default)
- **MS-NLMP** — NTLM authentication
- **MS-DCERPC, MS-RPCE** — DCE/RPC over named pipes carried inside SMB
- **MS-SAMR, MS-LSAD, MS-NETLOGON, MS-SRVS, MS-WKSSVC** — RPC interfaces
- **MS-RPRN, MS-PAR** — print-spooler RPC
- **MS-ADTS** — Active Directory Technical Specification
- **MS-DRSR** — Directory Replication
- **MS-KILE, MS-PAC** — Kerberos protocol extensions and Privilege Attribute
  Certificate
- **RFC 4178** (SPNEGO), **RFC 4120** (Kerberos 5), **RFC 4511** (LDAPv3),
  **RFC 1001/1002** (NetBIOS-over-TCP/IP), **RFC 2307** (NIS schema),
  **RFC 1035** (DNS)

## Functional scope (must)

### File server
- **`smbd`** daemon listening on TCP/445. SMB 2.1, 3.0, 3.0.2 and 3.1.1
  dialects negotiable; SMB1 is supported only behind an explicit
  `--legacy-smb1` flag, off by default.
- SMB3 packet **signing** (HMAC-SHA256 / AES-CMAC) and **encryption**
  (AES-128-CCM, AES-128-GCM, AES-256-GCM) per dialect.
- Tree connect, file open, read, write, set/get info, query directory,
  change-notify, locking, and oplocks/leases (level II, exclusive, batch,
  RWH leases).
- Pluggable VFS modules: at minimum a `default` module backed by the host
  filesystem, a `recycle` module diverting deletes, an `audit` module
  logging operations, and a `shadow_copy2` module exposing snapshots.
- Per-share configuration: read-only/writable, valid users, hosts allow/deny,
  veto files, dos charset, case sensitivity.

### Name resolution & browsing
- **`nmbd`** daemon: NBT name registration/release on UDP/137, datagrams on
  UDP/138, browser elections, master browser maintenance.
- WINS server mode (in-memory; persistence to a `tdb`-style store).

### ID mapping
- **`winbindd`** daemon: trust enumeration, SID↔UID/GID mapping (idmap_rid,
  idmap_autorid, idmap_ad), NSS handler protocol on a Unix socket.

### DCE/RPC
- DCE/RPC PDU framing (request, response, bind, alter_context, fault), NDR
  marshalling/unmarshalling.
- Implementations of: **SAMR** (account management), **LSARPC** (security
  policy), **NETLOGON** (secure channel + schannel sealing), **SRVSVC**
  (share enumeration), **WKSSVC** (workstation info), **SPOOLSS** (print
  spooler).

### Authentication
- **NTLMSSP**: NEGOTIATE/CHALLENGE/AUTHENTICATE messages, NTLMv2 response,
  Message Integrity Code, key derivation.
- **Kerberos 5**: AS, TGS, AP exchanges; AES-128-CTS-HMAC-SHA1-96 and
  AES-256-CTS-HMAC-SHA1-96 enctypes; keytab read/write; PAC issuance and
  validation.
- **GSSAPI/SPNEGO**: mechanism negotiation tokens, `mechListMIC`.
- **passdb**: local secrets store for standalone-server mode (tdb backend).

### Active Directory Domain Controller
- **`ldap-server`**: LDAPv3 on TCP/389 + LDAPS on TCP/636, with the AD schema
  preloaded. Supports `simple` and `SASL/GSSAPI` binds. Implements
  paged-results, sort, and persistent-search controls.
- **`kdc-server`**: KDC on UDP/88 + TCP/88, issuing TGTs with PACs.
- **`dns-server`**: AD-integrated DNS on UDP/53, with secure dynamic updates
  (RFC 3645 / GSS-TSIG).
- **`dirsync`**: DRSUAPI replication transport. A clearly-documented stub
  that records inbound replication requests and what it would have applied
  is acceptable — full multi-master semantics are not required.

### Print server
- **SPOOLSS** RPC dispatched over named pipes; backend shells out to a
  CUPS-compatible spooler.

### Admin & client tools (CLI binaries)
- `samba-tool` — domain provision, user/group management, FSMO transfer,
  GPO basics, DNS record management.
- `smbclient` — interactive SMB client with FTP-style commands.
- `nmblookup`, `wbinfo`, `pdbedit`, `net`, `testparm`.

## Out of scope (won't)

- Migration tools targeting other vendors' AD implementations (`samba-tool
  domain migrate`).
- Group Policy template *authoring* — GPO consumption is in scope; authoring
  is not.
- VFS modules targeting clustered filesystems (CTDB, GlusterFS).
- macOS-specific extensions (`fruit` VFS module).
- IPv6 multicast for NetBIOS name resolution (NBT is IPv4-only by spec; do
  not invent an IPv6 variant).
- Fileserver clustering and SMB Multichannel — single-node only.

## Architecture (suggested workspace layout)

A workspace with one crate per major subsystem. Suggested top-level crates:

**Support layer**
- `tdb` — trivial database (single-file key/value, byte-level locking).
- `tevent` — event-loop façade. May re-export a curated subset of `tokio`.
- `talloc` — hierarchical allocator façade. In safe Rust this is mostly
  a typed `Arena` and lifetime documentation; keep the crate so the layout
  matches Samba's own tree.
- `smb-config` — `smb.conf` parser, runtime config snapshot, reload.
- `ndr` — IDL → wire marshalling/unmarshalling primitives.

**Protocol framing**
- `smb-protocol` — SMB2/3 packet types, parser, encoder, capability
  negotiation, transport-agnostic.
- `smb1-protocol` — SMB1/CIFS framing (gated, optional).
- `smb-transport` — connection plumbing: NetBIOS-over-TCP framing
  (RFC 1001/1002), direct TCP, SMB Direct stub.
- `dcerpc` — DCE/RPC PDU framing, association groups, bind/alter/call.

**Per-RPC interfaces** (all depend on `ndr` + `dcerpc`)
- `rpc-samr`, `rpc-lsad`, `rpc-srvs`, `rpc-wkssvc`, `rpc-netlogon`,
  `rpc-rprn`, `rpc-drsuapi`.

**Authentication**
- `ntlmssp` — NTLMSSP message types, MIC, key derivation.
- `kerberos` — KRB5 ASN.1 PDU types, AS/TGS/AP exchanges, keytab, PAC.
- `gssapi` — SPNEGO negotiation glue.
- `auth-passdb` — local password database (tdb-backed).

**Storage / sharing**
- `vfs` — VFS trait and built-in modules (`default`, `recycle`, `audit`,
  `shadow_copy2`).
- `share` — share table, ACL evaluation, oplocks/leases, change-notify.

**AD-DC stack**
- `ldap-server`, `kdc-server`, `dns-server`, `dirsync`.

**Daemons (binaries)**
- `smbd`, `nmbd`, `winbindd`.

**Clients & admin tools (binaries)**
- `samba-tool`, `smbclient`, `nmblookup`, `wbinfo`, `pdbedit`, `net`,
  `testparm`.

**Black-box tests**
- `interop-tests` — integration tests that drive a running daemon over
  loopback. Gated behind `cargo test -- --ignored` if a fixture share on
  disk is required.

The dependency DAG threads through the layout: every RPC crate depends on
`ndr` + `dcerpc`; every daemon depends on `smb-protocol` + `vfs` + `share` +
the appropriate auth crates; the AD-DC crates depend on `kerberos` +
`ldap-server`. This is exactly the kind of broad, deep, cross-crate graph
the workspace mode is designed to exercise.

## Constraints

1. **Workspace layout.** Use the workspace mode. Keep each leaf node ≤1000
   LOC of Rust source; large protocol crates (e.g. `smb-protocol`,
   `kerberos`) may decompose internally into submodules.
2. **TLS via `rustls` / `aws-lc-rs`.** No OpenSSL or GnuTLS. Where the
   protocol mandates a specific cipher (e.g. SMB3 AES-128-CCM), implement
   it via `aws-lc-rs` primitives.
3. **No `unsafe` outside FFI shims.** None of the protocol or RPC code
   needs `unsafe`; if a crate reaches for it, that's a smell.
4. **Per-crate unit tests are mandatory.** Wire-format crates additionally
   carry round-trip tests against captured packet fixtures (hex strings
   are fine — no binary blobs in the repo).
5. **No live network in the default test profile.** Integration tests that
   need loopback are `#[ignore]` and run via `cargo test -- --ignored`.
6. **Documentation.** Every crate's `lib.rs` carries a doc comment that
   names the MS-* / RFC reference(s) it implements, and a one-paragraph
   summary of what subset is in scope.
7. **No reach-around imports.** A crate may only depend on crates that are
   declared in the `deps` of its node.
8. **Stable serde.** Anything that crosses a process boundary or hits disk
   uses serde with stable field names — schema changes are explicit.

## Suggested decomposition / build order

The agent should decompose hierarchically. A reasonable bottom-up order
(leaves first; the engine's bottom-up `Impl` ordering will follow the dep
graph):

1. **Support primitives**: `tdb`, `tevent`, `talloc`, `smb-config`, `ndr`.
2. **Protocol framing**: `smb-protocol`, `smb-transport`, `dcerpc`.
3. **Auth**: `ntlmssp` → `kerberos` → `gssapi` → `auth-passdb`.
4. **Storage**: `vfs` (default + the 3 built-in modules) → `share`.
5. **Per-RPC interfaces** (parallelizable): `rpc-srvs`, `rpc-wkssvc`,
   `rpc-samr`, `rpc-lsad`, `rpc-netlogon`, `rpc-rprn`, `rpc-drsuapi`.
6. **File server daemon**: `smbd`.
7. **Other daemons**: `nmbd`, `winbindd`.
8. **Clients & admin tools**: `samba-tool`, `smbclient`, the small CLIs.
9. **AD-DC stack**: `ldap-server` → `kdc-server` → `dns-server` →
   `dirsync`.

At every level, prefer parallel siblings over a long chain; the scheduler
will run independent sibling stages in parallel up to `max_parallel_tasks`.

## Acceptance criteria

A reviewer who knows Samba should be able to:

- Read each crate's `lib.rs` doc comment and recognize the spec(s) it
  implements.
- Find `cargo test` passing in every crate.
- Run `cargo run --bin smbd -- --foreground --config example/smb.conf` and
  connect from a Linux client (`smbclient //127.0.0.1/share -U user`),
  producing a session that authenticates, lists files, and reads one.
- Run `cargo run --bin samba-tool -- domain provision --realm=EXAMPLE.LOCAL
  --domain=EXAMPLE` and observe the LDAP/KDC/DNS daemons starting against
  the provisioned database.

The system does **not** need to interoperate with real Windows clients to
pass review — a thoughtful reimplementation that matches the public specs
and is internally consistent is the bar. Where a spec demands behavior
that is impractical to implement standalone (full DRSUAPI replication
semantics, kernel-level oplock break delivery), a clearly-commented stub
that records what it received and what it would have done is acceptable,
provided the stub is honest about its limitations in a `lib.rs` doc
comment.
