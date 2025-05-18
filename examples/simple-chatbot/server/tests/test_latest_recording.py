import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from server import app, daily_helpers

client = TestClient(app)


@pytest.fixture
def mock_setup():
    """Set up the base mocks but allow each test to configure responses."""
    with patch("aiohttp.ClientSession") as mock_session:
        # Create mock objects that tests can configure
        mock_session_instance = mock_session.return_value
        mock_session_instance.get = MagicMock()
        
        # Setup daily helpers
        daily_helpers["rest"] = AsyncMock()
        daily_helpers["rest"].daily_api_url = "https://api.daily.co/v1"
        daily_helpers["rest"].daily_api_key = "test_api_key"
        daily_helpers["rest"].aiohttp_session = mock_session_instance
        
        yield mock_session_instance


@pytest.mark.asyncio
async def test_get_latest_recording_success(mock_setup):
    """Test successful retrieval of the latest recording."""
    # Create mock responses for this test
    mock_recordings_response = AsyncMock()
    mock_recordings_response.status = 200
    mock_recordings_response.json.return_value = {
        "total_count": 2,
        "data": [
            {"id": "rec_123", "start_ts": 1000, "room_name": "test-room1"},
            {"id": "rec_456", "start_ts": 2000, "room_name": "test-room2"}
        ]
    }
    
    mock_access_link_response = AsyncMock()
    mock_access_link_response.status = 200
    mock_access_link_response.json.return_value = {
        "download_link": "https://example.com/recording.mp4"
    }
    
    # Configure mock to return our responses
    mock_context_managers = [MagicMock(), MagicMock()]
    mock_context_managers[0].__aenter__.return_value = mock_recordings_response
    mock_context_managers[1].__aenter__.return_value = mock_access_link_response
    
    # Make the get method return different context managers on successive calls
    mock_setup.get.side_effect = mock_context_managers
    
    # Make the request to our endpoint
    response = client.get("/latest_recording/")
    
    # Verify response
    assert response.status_code == 200
    assert response.json() == {"download_link": "https://example.com/recording.mp4"}
    
    # Verify the API calls were made correctly
    calls = mock_setup.get.call_args_list
    assert len(calls) == 2
    
    # Check first call (to get recordings)
    recordings_call = calls[0]
    assert recordings_call[0][0] == "https://api.daily.co/v1/recordings"
    assert recordings_call[1]["headers"] == {"Authorization": "Bearer test_api_key"}
    
    # Check second call (to get access link)
    access_link_call = calls[1]
    assert access_link_call[0][0] == "https://api.daily.co/v1/recordings/rec_456/access-link"
    assert access_link_call[1]["headers"] == {"Authorization": "Bearer test_api_key"}


@pytest.mark.asyncio
async def test_get_latest_recording_no_recordings(mock_setup):
    """Test when no recordings are available."""
    # Setup empty recordings response
    mock_empty_response = AsyncMock()
    mock_empty_response.status = 200
    mock_empty_response.json.return_value = {
        "total_count": 0,
        "data": []
    }
    
    # Setup the mock to return our empty response
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__.return_value = mock_empty_response
    mock_setup.get.return_value = mock_context_manager
    
    # Make the request to our endpoint
    response = client.get("/latest_recording/")
    
    # Verify response
    assert response.status_code == 404
    assert response.json() == {"detail": "No recordings found"}


@pytest.mark.asyncio
async def test_get_latest_recording_api_error(mock_setup):
    """Test handling of API errors."""
    # Setup error response
    mock_error_response = AsyncMock()
    mock_error_response.status = 500
    # API error responses may need to trigger an exception or error
    mock_error_response.json.side_effect = Exception("API error")
    mock_error_response.text.return_value = "Server error"
    
    # Setup the mock to return our error response
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__.return_value = mock_error_response
    mock_setup.get.return_value = mock_context_manager
    
    # Make the request to our endpoint
    response = client.get("/latest_recording/")
    
    # Verify response
    assert response.status_code == 500
    assert "Failed to get recordings" in response.json()["detail"] 