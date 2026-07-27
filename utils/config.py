# utils/config.py
"""config.yaml 로딩 및 CLI 오버라이드 유틸리티.

모든 하이퍼파라미터는 config.yaml 에 있고, 실험별 변경은
`--set key.path=value` 또는 main.py 의 전용 플래그로만 수행한다.
"""

import copy
import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "config.yaml")


class Config(dict):
    """점(.) 접근을 지원하는 중첩 dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict):
            return Config({k: Config._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    @classmethod
    def from_dict(cls, data):
        return cls._wrap(dict(data))

    def to_dict(self):
        out = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted, value):
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config._wrap(value)
        return self

    def copy(self):
        return Config.from_dict(copy.deepcopy(self.to_dict()))

    def dump(self, path):
        """현재 설정을 YAML 로 저장한다 (실험 기록용)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
        return path


def _parse_scalar(text):
    """`--set` 로 들어온 문자열을 YAML 규칙으로 파싱 (true/1e-4/[1,2] 등 지원)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path=None, overrides=None):
    """config.yaml 로드 후 오버라이드 적용.

    overrides: ["train.batch_size=4", "teacher.name=sd15"] 형태의 리스트
    """
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config.from_dict(data)
    cfg.set_path("_config_path", os.path.abspath(path))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식이 잘못되었습니다 (key=value): {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if cfg.get_path(key, _MISSING) is _MISSING:
            raise KeyError(f"config 에 존재하지 않는 키입니다: {key}")
        cfg.set_path(key, _parse_scalar(raw.strip()))
    return cfg


class _Missing:
    pass


_MISSING = _Missing()
