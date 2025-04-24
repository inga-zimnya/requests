import pytest
import requests
from typing import Tuple

# Assume actions.py defines ActionClient
from actions import ActionClient
from main import fetch_game_state

class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.HTTPError(f"Status code: {self.status_code}")

# Helper to patch requests.post
@pytest.fixture(autouse=True)
def patch_requests_post(monkeypatch):
    calls = {}
    def fake_post(url, json, timeout):
        calls['url'] = url
        calls['json'] = json
        calls['timeout'] = timeout
        # Default success
        return DummyResponse({'result': True}, 200)
    monkeypatch.setattr(requests, 'post', fake_post)
    return calls


@pytest.fixture
def action_client():
    client = ActionClient('http://127.0.0.1:15702/')
    # Pre-populate with test positions
    client.update_pickup_ids([(1, 1), (2, 2), (3, 3)])
    return client


def test_pickup_by_position(action_client, requests_mock):
    # Mock the server response
    requests_mock.post('http://127.0.0.1:15702/bevy/pickup_item', status_code=200)

    # Position (1,1) should map to ID 1 (first in our test list)
    result = action_client.send_action(
        entity_id=123,
        action_type='pickup',
        action_data={'position': (1, 1)}
    )

    assert result is True
    assert requests_mock.last_request.json() == {
        'entity': 123,
        'item_id': 1  # Should be first assigned ID
    }

# Tests for send_action
@ pytest.mark.parametrize("action_type, method, params", [
    ('move', 'bevy/move_entity', {'entity': 123, 'direction': [1, 2], 'speed': 3.4}),
    ('shoot', 'bevy/shoot', {'entity': 456, 'direction': [2, 1], 'force': 2.5}),
    ('kick', 'bevy/kick', {'entity': 789, 'direction': [0, 1]}),
    ('pickup', 'bevy/pickup_item', {'entity': 321, 'item_id': 999}),
    ('reload', 'bevy/reload_weapon', {'entity': 555}),
])
def test_send_action_success(patch_requests_post, action_type, method, params):
    client = ActionClient('http://127.0.0.1:15702/')
    result = client.send_action(params['entity'], action_type,
                                {k: v for k, v in params.items() if k != 'entity'})
    assert result is True
    assert patch_requests_post['url'] == 'http://127.0.0.1:15702/'
    assert patch_requests_post['timeout'] == 1.0
    payload = patch_requests_post['json']
    assert payload['method'] == method
    assert payload['params'] == params


def test_send_action_invalid_type():
    client = ActionClient('http://127.0.0.1:15702/')
    with pytest.raises(ValueError):
        client.send_action(1, 'fly')


def test_send_action_http_error(monkeypatch):
    # Simulate HTTP error
    def fake_post_error(url, json, timeout):
        return DummyResponse({'error': 'fail'}, status_code=500)
    monkeypatch.setattr(requests, 'post', fake_post_error)
    client = ActionClient('http://127.0.0.1:15702/')
    assert client.send_action(1, 'move', {'direction': [0,0]}) is False

# Tests for convenience methods
class FakeState:
    def __init__(self, players):
        self.players = players

@pytest.fixture
def fake_state(monkeypatch):
    def _make_state(players):
        state = FakeState(players)
        monkeypatch.setattr('main.fetch_game_state', lambda: state)
        return state
    return _make_state


def test_move_to_success(patch_requests_post, fake_state):
    # set current position
    entity_id = 10
    fake_state({entity_id: {'position': (0, 0)}})
    client = ActionClient('http://127.0.0.1:15702/')
    # move to (3,4)
    result = client.move_to(entity_id, (3, 4), speed=2.0)
    assert result is True
    # direction should be [3,4]
    assert patch_requests_post['json']['params']['direction'] == [3, 4]
    assert patch_requests_post['json']['params']['speed'] == 2.0


def test_move_to_no_state(monkeypatch):
    # fetch_game_state returns None
    monkeypatch.setattr('main.fetch_game_state', lambda: None)
    client = ActionClient('http://127.0.0.1:15702/')
    assert client.move_to(1, (1,1)) is False


def test_move_to_invalid_entity(monkeypatch):
    # state without entity
    monkeypatch.setattr('main.fetch_game_state', lambda: FakeState({}))
    client = ActionClient('http://127.0.0.1:15702/')
    assert client.move_to(1, (1,1)) is False

@pytest.mark.parametrize("method_name, target, force_key", [
    ('shoot_at', (5,5), 'force'),
    ('kick_at', (2,3), None)
])
def test_action_convenience(monkeypatch, patch_requests_post, fake_state, method_name, target, force_key):
    entity_id = 7
    fake_state({entity_id: {'position': (1, 1)}})
    client = ActionClient('http://127.0.0.1:15702/')
    # call method dynamically
    func = getattr(client, method_name)
    if force_key:
        result = func(entity_id, target, force=4.5)
    else:
        result = func(entity_id, target)
    assert result is True
    params = patch_requests_post['json']['params']
    # direction = target - current
    expected_dir = [target[0] - 1, target[1] - 1]
    assert params['direction'] == expected_dir
    if force_key:
        assert params[force_key] == 4.5


def test_shoot_at_no_state(monkeypatch):
    monkeypatch.setattr('main.fetch_game_state', lambda: None)
    client = ActionClient('http://127.0.0.1:15702/')
    assert client.shoot_at(1, (1,1)) is False


def test_kick_at_invalid_entity(monkeypatch):
    monkeypatch.setattr('main.fetch_game_state', lambda: FakeState({}))
    client = ActionClient('http://127.0.0.1:15702/')
    assert client.kick_at(1, (0,0)) is False
