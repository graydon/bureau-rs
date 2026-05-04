# Reimplement Samba in Rust

Build a Rust workspace that reimplements a substantial subset of the
Samba server suite (the open-source Windows-file-sharing implementation
for Unix). The feature list below transcribes Samba's own feature
inventory.

## Why this is worth doing

Samba is the canonical interoperability server for Windows file sharing
on Unix — three decades of accumulated protocol work across SMB/CIFS,
NetBIOS, DCE/RPC, NTLM, Kerberos, and the AD-DC stack. The protocols
are public and well-documented; the surface is large but well-bounded;
a memory-safe Rust implementation of a security-critical server is
genuinely useful. It also makes a substantial test case for
hierarchical decomposition: the feature list naturally splits into many
subsystems that share deps (auth, RPC framing, transport).

## In scope

A daemon set comparable to a Samba 4 standalone or domain-member
install:

- **SMB/CIFS file serving** to Windows clients. Negotiable dialects
  SMB 2.0, 2.1, 3.0, 3.0.2, 3.1.1; SMB 1 (CIFS) optional and off by
  default. SMB3 signing (HMAC-SHA256, AES-CMAC) and encryption
  (AES-128-CCM, AES-128-GCM, AES-256-GCM).
- **NetBIOS name service** over UDP/137,138 — name registration,
  resolution, master-browser elections. WINS-server mode.
- **ID mapping** — resolve Windows users / groups (SID ↔ UID/GID) for
  NSS, against a local DB or a remote DC.
- **DCE/RPC** carried inside SMB named pipes, exposing the interfaces
  Samba's tools and Windows clients expect: SAMR (account management),
  LSARPC (security policy), SRVSVC (share enumeration), WKSSVC
  (workstation info), NETLOGON (secure channel), SPOOLSS (print
  spooler), DRSUAPI (directory replication).
- **Authentication**: NTLMv2 via NTLMSSP, Kerberos 5 (initiator and
  acceptor) tunnelled through GSSAPI / SPNEGO, with PAC issuance and
  validation. Local password DB for standalone-server mode.
- **Active Directory Domain Controller mode**: LDAPv3 server (TCP 389
  / 636), KDC (UDP/TCP 88), AD-integrated DNS server with secure
  dynamic updates, and inter-DC replication. Honest stubs are
  acceptable where a full implementation would be impractical.
- **Pluggable VFS modules**, at minimum: a default backend over the
  local filesystem; a recycle-bin module diverting deletes; an audit
  module logging operations; a shadow-copy module exposing filesystem
  snapshots via the Windows Previous Versions UI.
- **Share layer**: per-share configuration, ACL evaluation, oplocks /
  leases, change-notify.
- **Print serving** via SPOOLSS, backed by a CUPS-compatible spooler.
- **Operator tooling**: server daemons (`smbd`, `nmbd`, `winbindd`);
  admin CLIs (`samba-tool`, `net`, `testparm`, `pdbedit`); client and
  diagnostic tools (`smbclient`, `nmblookup`, `wbinfo`).

## Out of scope

- Migration tools targeting other vendors' AD implementations.
- Group Policy template authoring (consumption is in scope).
- VFS modules for clustered filesystems (CTDB, GlusterFS).
- macOS-specific extensions (the `fruit` VFS module).
- Fileserver clustering and SMB Multichannel — single-node only.

## What "done" looks like

A reviewer who knows Samba should be able to:

- Read each crate's `lib.rs` doc comment and recognize what subset of
  the public Microsoft / IETF specs it implements.
- Run `cargo test` across the workspace, green.
- Connect from a stock Linux client (`smbclient //127.0.0.1/share
  -U user`) and authenticate, list, and read a file from a fixture
  share.
- Run `samba-tool domain provision` and watch the AD-DC daemons start
  against the provisioned database.

The system does not need to interoperate with real Windows clients — a
thoughtful, internally-consistent implementation matching the public
specs is the bar.
