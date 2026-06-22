"""
YouTube Data API v3 uploader.

Replaces the deprecated Selenium upload path. Uses the OAuth installed-app flow
(``InstalledAppFlow.run_local_server``) and caches the resulting token locally so
subsequent uploads run without re-authorising.

The Selenium upload path still exists in ``classes/YouTube.py`` but is gated
behind the ``use_selenium_upload`` config flag (default false).

Google client libraries are imported lazily so this module can be imported (and
the rest of the app can run) even when the optional dependencies are absent.
"""

import os
from typing import Optional, Sequence

from config import (
    get_youtube_category_id,
    get_youtube_client_secrets_file,
    get_youtube_contains_synthetic_media,
    get_youtube_default_tags,
    get_youtube_privacy_status,
    get_youtube_token_file,
    get_is_for_kids,
)
from status import info, success

# Uploading is all we need; this is the minimal scope.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class YouTubeUploadError(RuntimeError):
    """Raised when an upload cannot be completed."""


def _load_credentials(
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
):
    """
    Loads cached OAuth credentials, refreshing or running the installed-app
    flow as needed, and persists the (refreshed) token.

    Returns:
        credentials: An authorized google.oauth2.credentials.Credentials.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise YouTubeUploadError(
            "Google API client libraries are not installed. Install "
            "google-api-python-client, google-auth-oauthlib and "
            "google-auth-httplib2."
        ) from exc

    client_secrets_file = client_secrets_file or get_youtube_client_secrets_file()
    token_file = token_file or get_youtube_token_file()

    credentials = None
    if os.path.exists(token_file):
        credentials = Credentials.from_authorized_user_file(token_file, SCOPES)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    else:
        if not os.path.exists(client_secrets_file):
            raise YouTubeUploadError(
                f"OAuth client secrets file not found: {client_secrets_file}. "
                "Download it from the Google Cloud console (OAuth client, "
                "type 'Desktop app') and point youtube.client_secrets_file at it."
            )
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
        credentials = flow.run_local_server(port=0)

    token_parent = os.path.dirname(token_file)
    if token_parent:
        os.makedirs(token_parent, exist_ok=True)
    with open(token_file, "w", encoding="utf-8") as handle:
        handle.write(credentials.to_json())

    return credentials


def _build_status(
    privacy_status: str,
    made_for_kids: bool,
    contains_synthetic_media: bool,
) -> dict:
    """
    Builds the ``status`` portion of the videos.insert body, including the
    made-for-kids declaration and the AI/altered-content disclosure.
    """
    status = {
        "privacyStatus": privacy_status,
        "selfDeclaredMadeForKids": made_for_kids,
    }
    # YouTube's altered/synthetic content disclosure. Declaring it is the
    # responsible default for AI-generated content.
    if contains_synthetic_media:
        status["containsSyntheticMedia"] = True
    return status


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: Optional[Sequence[str]] = None,
    category_id: Optional[str] = None,
    privacy_status: Optional[str] = None,
    made_for_kids: Optional[bool] = None,
    contains_synthetic_media: Optional[bool] = None,
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
) -> str:
    """
    Uploads a video via the YouTube Data API (videos.insert) and returns its id.

    Args:
        video_path (str): Path to the MP4 to upload.
        title (str): Video title.
        description (str): Video description.
        tags (Sequence[str] | None): Tags. Defaults to youtube.default_tags.
        category_id (str | None): Category id. Defaults to config.
        privacy_status (str | None): Privacy status. Defaults to config.
        made_for_kids (bool | None): Made-for-kids declaration. Defaults to the
            top-level ``is_for_kids`` config flag.
        contains_synthetic_media (bool | None): AI/altered content disclosure.
            Defaults to config (true for this engine).
        client_secrets_file (str | None): Override for OAuth client secrets.
        token_file (str | None): Override for the cached token path.

    Returns:
        video_id (str): The uploaded video's id.

    Raises:
        YouTubeUploadError: If the upload cannot be completed.
    """
    if not os.path.exists(video_path):
        raise YouTubeUploadError(f"Video file does not exist: {video_path}")

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise YouTubeUploadError(
            "google-api-python-client is not installed."
        ) from exc

    if tags is None:
        tags = get_youtube_default_tags()
    if category_id is None:
        category_id = get_youtube_category_id()
    if privacy_status is None:
        privacy_status = get_youtube_privacy_status()
    if made_for_kids is None:
        made_for_kids = get_is_for_kids()
    if contains_synthetic_media is None:
        contains_synthetic_media = get_youtube_contains_synthetic_media()

    credentials = _load_credentials(client_secrets_file, token_file)
    youtube = build("youtube", "v3", credentials=credentials)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": list(tags),
            "categoryId": str(category_id),
        },
        "status": _build_status(
            privacy_status, bool(made_for_kids), bool(contains_synthetic_media)
        ),
    }

    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    info(f"Uploading to YouTube via Data API: {title}")
    response = None
    try:
        while response is None:
            _status, response = request.next_chunk()
    except HttpError as exc:  # pragma: no cover - network dependent
        raise YouTubeUploadError(f"YouTube API upload failed: {exc}") from exc

    video_id = response.get("id")
    if not video_id:
        raise YouTubeUploadError(
            f"YouTube API did not return a video id. Response: {response}"
        )

    success(f"Uploaded YouTube video id: {video_id}")
    return video_id
