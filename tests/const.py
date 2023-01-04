import os

path_to_test_dir = os.path.dirname(os.path.abspath(__file__))


def normalize_path(rel_path):
    return os.path.join(path_to_test_dir, rel_path)


VALIDATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["name", "x", "y"],
    },
}
