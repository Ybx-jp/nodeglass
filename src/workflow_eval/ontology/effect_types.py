"""EffectType and EffectTarget enums."""

from enum import StrEnum


class EffectType(StrEnum):
    """Classification of an operation's side-effect behavior."""

    PURE = "pure"
    STATEFUL = "stateful"
    EXTERNAL = "external"
    IRREVERSIBLE = "irreversible"


class EffectTarget(StrEnum):
    """What system boundary an operation touches."""

    FILESYSTEM = "filesystem"
    NETWORK = "network"
    DATABASE = "database"
    MEMORY = "memory"
    USER_FACING = "user_facing"
    CREDENTIALS = "credentials"
    SYSTEM_CONFIG = "system_config"
