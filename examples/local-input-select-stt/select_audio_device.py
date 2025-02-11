from typing import List, Optional, Tuple

import pyaudio
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Label, ListItem, ListView, Select
from textual.widgets.option_list import Option

# ─── DATA MODELS ───────────────────────────────────────────────────────────────


class HostApi(BaseModel):
    index: int
    struct_version: int = Field(..., alias="structVersion")
    type: int
    name: str
    device_count: int = Field(..., alias="deviceCount")
    default_input_device: int = Field(..., alias="defaultInputDevice")
    default_output_device: int = Field(..., alias="defaultOutputDevice")


class AudioDevice(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    index: int
    struct_version: int = Field(..., alias="structVersion")
    name: str
    host_api: int = Field(..., alias="hostApi")
    max_input_channels: int = Field(..., alias="maxInputChannels")
    max_output_channels: int = Field(..., alias="maxOutputChannels")
    default_low_input_latency: float = Field(..., alias="defaultLowInputLatency")
    default_low_output_latency: float = Field(..., alias="defaultLowOutputLatency")
    default_high_input_latency: float = Field(..., alias="defaultHighInputLatency")
    default_high_output_latency: float = Field(..., alias="defaultHighOutputLatency")
    default_sample_rate: float = Field(..., alias="defaultSampleRate")


# ─── SETTINGS MODEL ───────────────────────────────────────────────────────────


class AudioSettings(BaseSettings):  # to save settings to a file
    host_api: Optional[int] = None
    input_device: Optional[AudioDevice] = None
    output_device: Optional[AudioDevice] = None

    class Config:
        env_file = "settings.env"  # or adjust as needed

    def save_to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))


# ─── TEXTUAL APP ──────────────────────────────────────────────────────────────


class AudioDeviceSelectorApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #container {
        width: 80%;
        border: round green;
        padding: 1 2;
    }
    """

    def __init__(
        self,
        default_host_api: Optional[int] = None,
        default_input_device: Optional[AudioDevice] = None,
        default_output_device: Optional[AudioDevice] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Save defaults passed from settings.
        self.default_host_api: Optional[int] = default_host_api
        self.default_input_device: Optional[AudioDevice] = default_input_device
        self.default_output_device: Optional[AudioDevice] = default_output_device

        self.pyaudio_instance = pyaudio.PyAudio()

        # Static datastructures: host APIs and devices as well‐typed models.
        self.host_apis: List[HostApi] = []
        self.current_host_api: Optional[int] = None

        self.all_input_devices: List[AudioDevice] = []
        self.all_output_devices: List[AudioDevice] = []
        self.input_devices: List[AudioDevice] = []
        self.output_devices: List[AudioDevice] = []

        # Stage management: first select input, then output.
        self.stage: str = "input"
        self.selected_input_device: Optional[AudioDevice] = None
        self.selected_output_device: Optional[AudioDevice] = None
        host_api_count: int = self.pyaudio_instance.get_host_api_count()
        for i in range(host_api_count):
            raw_api = self.pyaudio_instance.get_host_api_info_by_index(i)
            # Inject the index (if not already present)
            raw_api["index"] = i
            try:
                api = HostApi.parse_obj(raw_api)
                self.host_apis.append(api)
            except Exception as e:
                # Skip APIs that don't conform.
                continue

    def compose(self) -> ComposeResult:
        options: List[Tuple[str, Option]] = [
            (
                api.name,
                Option(
                    prompt=str(api.name) if api.name else f"Host API {api.index}",
                    id=str(api.index),
                ),
            )
            for api in self.host_apis
        ]

        yield Header()

        yield Footer()
        with Container(id="container"):
            yield Label("Select Host API:", id="host-api-label")
            # Create the Select widget with no options initially.
            self.host_api_select: Select[HostApi] = Select(options=options, id="host-api-select")
            yield self.host_api_select
            self.prompt = Label("Select Input Audio Device:", id="prompt")
            yield self.prompt
            self.list_view = ListView(id="device-list")
            yield self.list_view

    def on_mount(self) -> None:
        # Populate host APIs from PyAudio.

        # Build the dropdown options.

        self.host_api_select.refresh()  # Force a redraw

        # Determine the default host API.
        if self.default_host_api is not None:
            self.current_host_api = self.default_host_api
        else:
            default_api_info = self.pyaudio_instance.get_default_host_api_info()
            self.current_host_api = default_api_info["index"]

        # Delay setting the dropdown's value until the widget is fully initialized.
        self.set_timer(
            0,
            lambda: setattr(self.host_api_select, "value", str(self.current_host_api)),
        )

        # Load all devices and parse them into AudioDevice objects.
        device_count: int = self.pyaudio_instance.get_device_count()
        for i in range(device_count):
            raw_device = self.pyaudio_instance.get_device_info_by_index(i)
            raw_device["index"] = i
            try:
                device = AudioDevice.parse_obj(raw_device)
            except Exception as e:
                # Skip devices missing required fields.
                continue
            if device.max_input_channels > 0:
                self.all_input_devices.append(device)
            if device.max_output_channels > 0:
                self.all_output_devices.append(device)

        self.filter_devices()
        self.populate_list(self.input_devices)
        if self.default_input_device:
            self._select_default_in_list(self.default_input_device)

    def filter_devices(self) -> None:
        """Filter devices based on the selected host API."""
        self.input_devices = [
            d for d in self.all_input_devices if d.host_api == self.current_host_api
        ]
        self.output_devices = [
            d for d in self.all_output_devices if d.host_api == self.current_host_api
        ]

    def populate_list(self, devices: List[AudioDevice]) -> None:
        """Populate the ListView with a list of AudioDevice objects."""
        self.list_view.clear()
        for dev in devices:
            item_text: str = f"{dev.name} (Index: {dev.index})"
            item = ListItem(Label(item_text))
            # Attach the AudioDevice instance to the widget.
            item.device_info = dev  # type: ignore
            self.list_view.append(item)

    def _select_default_in_list(self, default_device: AudioDevice) -> None:
        """Pre-select the default device if present in the current list."""
        for idx, item in enumerate(self.list_view.children):
            if hasattr(item, "device_info") and item.device_info.index == default_device.index:
                self.list_view.index = idx
                break

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle changes in the host API dropdown."""
        if event.select.id == "host-api-select":
            self.current_host_api = int(event.value.id)
            self.filter_devices()
            if self.stage == "input":
                self.populate_list(self.input_devices)
                if self.default_input_device:
                    self._select_default_in_list(self.default_input_device)
            elif self.stage == "output":
                self.populate_list(self.output_devices)
                if self.default_output_device:
                    self._select_default_in_list(self.default_output_device)

    async def on_list_view_selected(self, message: ListView.Selected) -> None:
        """Record device selection and switch stages."""
        selected_item = message.item
        device_info: AudioDevice = selected_item.device_info  # type: ignore
        if self.stage == "input":
            self.selected_input_device = device_info
            self.stage = "output"
            self.prompt.update("Select Output Audio Device:")
            self.populate_list(self.output_devices)
            if self.default_output_device:
                self._select_default_in_list(self.default_output_device)
        elif self.stage == "output":
            self.selected_output_device = device_info
            await self.action_quit()


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────


async def run_device_selector(
    default_host_api: Optional[int] = None,
    default_input_device: Optional[AudioDevice] = None,
    default_output_device: Optional[AudioDevice] = None,
) -> Tuple[AudioDevice, AudioDevice, int]:
    app = AudioDeviceSelectorApp(
        default_host_api=default_host_api,
        default_input_device=default_input_device,
        default_output_device=default_output_device,
    )
    await app.run_async()

    # The current_host_api is guaranteed to be set.
    return app.selected_input_device, app.selected_output_device, app.current_host_api  # type: ignore
