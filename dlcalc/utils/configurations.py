from enum import Enum


class ActivationCheckpointingType(Enum):
    NONE = 1
    SELECTIVE = 2
    SUPER_SELECTIVE = 3
    FULL = 4

    @staticmethod
    def from_str(str_key: str) -> "ActivationCheckpointingType":
        return {
            "none": ActivationCheckpointingType.NONE,
            "selective": ActivationCheckpointingType.SELECTIVE,
            "super-selective": ActivationCheckpointingType.SUPER_SELECTIVE,
            "full": ActivationCheckpointingType.FULL,
        }[str_key]
