import struct
from dataclasses import dataclass


try:
    import usb.core
    import usb.util
except ImportError:  # pragma: no cover
    usb = None


# USB ReSpeaker 4-Mic Array (USB) default IDs from official examples.
DEFAULT_VENDOR_ID = 0x2886
DEFAULT_PRODUCT_ID = 0x0018


TIMEOUT_MS = 1000
WRITE_TIMEOUT_MS = 100000


CTRL_IN = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE if usb else 0
CTRL_OUT = usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE if usb else 0


def _require_pyusb() -> None:
    if usb is None:
        raise ImportError(
            "pyusb is required for ReSpeaker built-in DOA/VAD.\n"
            "Install: pip install pyusb\n"
            "You may also need libusb (e.g. macOS: `brew install libusb`)."
        )


@dataclass(frozen=True)
class RespeakerIds:
    vendor_id: int = DEFAULT_VENDOR_ID
    product_id: int = DEFAULT_PRODUCT_ID


class RespeakerTuning:
    """Minimal ReSpeaker USB tuning reader (DOA + VAD).

    This implements the same vendor-control protocol used in:
    https://github.com/respeaker/usb_4_mic_array (tuning.py)
    """

    # From upstream PARAMETERS table.
    _ID_VAD = 19
    _ID_DOA = 21

    _OFFS_VOICEACTIVITY = 32  # VOICEACTIVITY (int)
    _OFFS_DOAANGLE = 0  # DOAANGLE (int, 0..359)
    _OFFS_GAMMAVAD_SR = 39  # GAMMAVAD_SR (float, VAD threshold in dB)

    def __init__(self, dev, timeout_ms: int = TIMEOUT_MS):
        _require_pyusb()
        self.dev = dev
        self.timeout_ms = int(timeout_ms)

    def close(self) -> None:
        try:
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass

    @staticmethod
    def find(ids: RespeakerIds = RespeakerIds()):
        _require_pyusb()
        vendor = int(ids.vendor_id)
        product = int(ids.product_id)
        dev = usb.core.find(idVendor=vendor, idProduct=product)
        if dev is None:
            # Some backends/environments fail filtered search while find_all still lists devices.
            # Fallback to manual matching for robustness.
            for item in usb.core.find(find_all=True) or []:
                try:
                    if int(item.idVendor) == vendor and int(item.idProduct) == product:
                        dev = item
                        break
                except Exception:
                    continue
        if dev is None:
            found = []
            for item in usb.core.find(find_all=True) or []:
                try:
                    found.append((int(item.idVendor), int(item.idProduct)))
                except Exception:
                    continue
            found_txt = ", ".join([f"0x{v:04x}:0x{p:04x}" for v, p in found]) if found else "none"
            raise RuntimeError(
                f"ReSpeaker USB device not found (vendor=0x{vendor:04x}, product=0x{product:04x}). "
                f"Visible USB IDs: {found_txt}"
            )
        return RespeakerTuning(dev)

    def _read_int(self, unit_id: int, offset: int) -> int:
        # Upstream protocol: cmd = 0x80|offset, int flag adds 0x40, length=8 (2 int32).
        cmd = (0x80 | int(offset)) | 0x40
        raw = self.dev.ctrl_transfer(CTRL_IN, 0, cmd, int(unit_id), 8, self.timeout_ms)
        buf = bytes(raw)
        v0, v1 = struct.unpack("ii", buf[:8])
        _ = v1
        return int(v0)

    def _read_float(self, unit_id: int, offset: int) -> float:
        cmd = (0x80 | int(offset))
        raw = self.dev.ctrl_transfer(CTRL_IN, 0, cmd, int(unit_id), 8, self.timeout_ms)
        buf = bytes(raw)
        v0, v1 = struct.unpack("ii", buf[:8])
        return float(v0) * (2.0 ** float(v1))

    def _write_float(self, unit_id: int, offset: int, value: float) -> None:
        # Match official usb_4_mic_array tuning.py:
        # payload = struct.pack('ifi', offset, float_value, 0)
        payload = struct.pack("ifi", int(offset), float(value), 0)
        self.dev.ctrl_transfer(
            CTRL_OUT,
            0,
            0,
            int(unit_id),
            payload,
            max(self.timeout_ms, WRITE_TIMEOUT_MS),
        )

    @property
    def direction_deg(self) -> int:
        """Board-reported DOA angle in degrees (0..359), per hardware overview."""
        return int(self._read_int(self._ID_DOA, self._OFFS_DOAANGLE))

    def is_voice(self) -> bool:
        """Board VAD gate (0/1)."""
        return bool(self._read_int(self._ID_VAD, self._OFFS_VOICEACTIVITY))

    def set_vad_threshold_db(self, threshold_db: float) -> None:
        """Optional: adjust board VAD threshold (dB)."""
        self._write_float(self._ID_VAD, self._OFFS_GAMMAVAD_SR, float(threshold_db))
