from enum import Enum


class ActivationCheckpointingType(Enum):
    NONE = 1
    SELECTIVE = 2
    FULL = 3

    @staticmethod
    def from_str(str_key: str) -> "ActivationCheckpointingType":
        return {
            "none": ActivationCheckpointingType.NONE,
            "selective": ActivationCheckpointingType.SELECTIVE,
            "full": ActivationCheckpointingType.FULL,
        }[str_key]
