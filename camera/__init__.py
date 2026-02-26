# camera package — 自動載入內建相機插件以觸發 @register_camera 註冊
from camera.base import (  # noqa: F401
    CAMERA_REGISTRY,
    CameraBase,
    CameraInfo,
    get_registered_cameras,
    register_camera,
)

# 載入內建驅動（觸發裝飾器註冊）
import camera.d435_stream   # noqa: F401
import camera.cognex_stream  # noqa: F401
