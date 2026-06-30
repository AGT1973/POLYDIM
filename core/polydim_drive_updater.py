#!/usr/bin/env python3
"""
polydim_drive_updater.py
========================
Updates a Google Drive file IN-PLACE — same fileId, new content.
Preserves all references (skills, CARPETAS, BACKLOG) pointing to that fileId.

The MCP connector only has create_file (new fileId every time).
This script uses Drive REST API files.update endpoint directly:
  PATCH https://www.googleapis.com/upload/drive/v3/files/{fileId}?uploadType=media
This patches content WITHOUT changing the fileId.

Usage:
  python polydim_drive_updater.py --file-id 1XYZ... --content-file new.md
  python polydim_drive_updater.py --file-id 1XYZ... --stdin
  python polydim_drive_updater.py --file-id 1XYZ... --content "text"
  python polydim_drive_updater.py --batch-json updates.json

Auth:
  OAuth2: credentials.json from Google Cloud Console (creates token.json on first run)
  Service account: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

Install:
  pip install google-api-python-client google-auth-oauthlib

Author:  ai.mpat.agt@gmail.com
Version: V1.0 — 2026-06-19
"""

import argparse, json, os, sys, io
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

SCOPES          = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE      = "token.json"
CREDENTIALS_FILE = "credentials.json"


def get_credentials():
    sa_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_key and Path(sa_key).exists():
        return service_account.Credentials.from_service_account_file(sa_key, scopes=SCOPES)
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_FILE).exists():
                print(f"ERROR: {CREDENTIALS_FILE} not found.\nDownload from Google Cloud Console > APIs & Services > Credentials.", file=sys.stderr)
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def update_file_in_place(file_id: str, new_content: str, mime_type: str = "text/plain", verbose: bool = True) -> dict:
    """Update Drive file content while preserving fileId."""
    if not GOOGLE_AVAILABLE:
        raise ImportError("pip install google-api-python-client google-auth-oauthlib")
    creds   = get_credentials()
    service = build("drive", "v3", credentials=creds)
    current = service.files().get(fileId=file_id, fields="id,name,mimeType,modifiedTime").execute()
    if verbose:
        print(f"Updating: {current['name']} ({file_id})")
    media = MediaIoBaseUpload(io.BytesIO(new_content.encode("utf-8")), mimetype=mime_type, resumable=False)
    updated = service.files().update(fileId=file_id, media_body=media, fields="id,name,mimeType,modifiedTime").execute()
    if verbose:
        print(f"  fileId preserved: {updated['id'] == file_id}  modified: {updated.get('modifiedTime','?')}")
    return updated


def batch_update(updates: list, verbose: bool = True) -> list:
    """Update multiple files. updates = [{file_id, content, mime_type?}]"""
    return [update_file_in_place(u["file_id"], u["content"], u.get("mime_type","text/plain"), verbose) for u in updates]


def main():
    parser = argparse.ArgumentParser(description="Update Drive file in-place (preserves fileId).")
    parser.add_argument("--file-id",      help="Drive fileId to update")
    parser.add_argument("--content",      help="New content as inline string")
    parser.add_argument("--content-file", help="Path to file with new content")
    parser.add_argument("--stdin",        action="store_true")
    parser.add_argument("--mime-type",    default="text/plain")
    parser.add_argument("--batch-json",   help="JSON file with [{file_id,content}]")
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()
    if not GOOGLE_AVAILABLE:
        print("ERROR: pip install google-api-python-client google-auth-oauthlib", file=sys.stderr); sys.exit(1)
    if args.batch_json:
        results = batch_update(json.loads(Path(args.batch_json).read_text()), not args.quiet)
        if not args.quiet: print(json.dumps([{"id":r["id"],"name":r["name"]} for r in results], indent=2))
        return
    if not args.file_id:
        parser.error("--file-id required for single-file update")
    if args.stdin:            content = sys.stdin.read()
    elif args.content_file:   content = Path(args.content_file).read_text("utf-8")
    elif args.content:        content = args.content
    else:                     parser.error("one of --content, --content-file, --stdin required")
    r = update_file_in_place(args.file_id, content, args.mime_type, not args.quiet)
    if not args.quiet: print(json.dumps({"id":r["id"],"name":r["name"]}, indent=2))

if __name__ == "__main__":
    main()
