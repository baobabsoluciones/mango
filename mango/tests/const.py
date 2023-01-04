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
