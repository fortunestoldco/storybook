class State:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def remove(self, key):
        if key in self.data:
            del self.data[key]

    def initialize_state(self, initial_data):
        self.data = initial_data

    def to_dict(self):
        return self.data

    @staticmethod
    def from_dict(state_dict):
        state = State()
        state.data = state_dict
        return state

class ProjectState(State):
    def __init__(self):
        super().__init__()
        self.data = {
            "current_state": "BRAINSTORMING",
            "novel_id": None,
            "author": None,
            "research_data": {},
            "writing_data": {},
            "publishing_data": {},
            "feedback": {},
            "completed": False
        }
