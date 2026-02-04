pub mod fs;
pub mod webgl;

mod keycodes;

use std::{
    cell::RefCell,
    collections::VecDeque,
    path::PathBuf,
    sync::{mpsc::Receiver, Mutex, OnceLock},
    thread_local,
};

use crate::{
    event::EventHandler,
    native::{NativeDisplayData, Request},
    KeyCode, KeyMods, MouseButton, TouchPhase,
};

// fn dropped_file_count(&mut self) -> usize {
//     self.dropped_files.bytes.len()
// }
// fn dropped_file_bytes(&mut self, index: usize) -> Option<Vec<u8>> {
//     self.dropped_files.bytes.get(index).cloned()
// }
// fn dropped_file_path(&mut self, index: usize) -> Option<PathBuf> {
//     self.dropped_files.paths.get(index).cloned()
// }

/// Events that may be deferred if EVENT_HANDLER is already borrowed.
///
/// Browser events (touch, mouse, keyboard) can fire during frame execution,
/// causing RefCell re-entrancy panics. When this happens, events are queued
/// and processed at the start of the next frame.
#[derive(Debug, Clone)]
enum DeferredEvent {
    Touch { phase: TouchPhase, id: u64, x: f32, y: f32 },
    MouseMove { x: f32, y: f32 },
    RawMouseMove { dx: f32, dy: f32 },
    MouseDown { btn: MouseButton, x: f32, y: f32 },
    MouseUp { btn: MouseButton, x: f32, y: f32 },
    MouseWheel { dx: f32, dy: f32 },
    KeyDown { key: KeyCode, mods: KeyMods, repeat: bool },
    KeyUp { key: KeyCode, mods: KeyMods },
    CharEvent { character: char, mods: KeyMods, repeat: bool },
    Resize { width: f32, height: f32 },
    Focus { has_focus: bool },
    FilesDropped,
}

thread_local! {
    static EVENT_HANDLER: RefCell<Option<Box<dyn EventHandler>>> = RefCell::new(None);
    static REQUESTS: RefCell<Option<Receiver<Request>>> = const { RefCell::new(None) };
    /// Queue for events that couldn't be delivered due to RefCell borrow conflict.
    /// Drained at the start of each frame before update/draw.
    static DEFERRED_EVENTS: RefCell<VecDeque<DeferredEvent>> = const { RefCell::new(VecDeque::new()) };
}

fn tl_event_handler<T, F: FnOnce(&mut dyn EventHandler) -> T>(f: F) -> T {
    EVENT_HANDLER.with(|globals| {
        let mut globals = globals.borrow_mut();
        let globals: &mut Box<dyn EventHandler> = globals.as_mut().unwrap();
        f(&mut **globals)
    })
}

/// Try to execute event handler immediately, or defer if already borrowed.
///
/// Returns true if executed immediately, false if deferred.
fn try_event_handler_or_defer<F>(event: DeferredEvent, f: F) -> bool
where
    F: FnOnce(&mut dyn EventHandler),
{
    EVENT_HANDLER.with(|globals| {
        match globals.try_borrow_mut() {
            Ok(mut guard) => {
                if let Some(handler) = guard.as_mut() {
                    f(&mut **handler);
                }
                true
            }
            Err(_) => {
                // RefCell already borrowed (re-entrancy during frame execution)
                // Queue the event for processing at start of next frame
                DEFERRED_EVENTS.with(|q| {
                    q.borrow_mut().push_back(event);
                });
                false
            }
        }
    })
}

/// Process all deferred events. Called at the start of each frame.
fn drain_deferred_events() {
    let events: Vec<DeferredEvent> = DEFERRED_EVENTS.with(|q| {
        q.borrow_mut().drain(..).collect()
    });

    for event in events {
        tl_event_handler(|handler| {
            dispatch_deferred_event(handler, event);
        });
    }
}

/// Dispatch a deferred event to the event handler.
fn dispatch_deferred_event(handler: &mut dyn EventHandler, event: DeferredEvent) {
    match event {
        DeferredEvent::Touch { phase, id, x, y } => {
            handler.touch_event(phase, id, x, y);
        }
        DeferredEvent::MouseMove { x, y } => {
            handler.mouse_motion_event(x, y);
        }
        DeferredEvent::RawMouseMove { dx, dy } => {
            handler.raw_mouse_motion(dx, dy);
        }
        DeferredEvent::MouseDown { btn, x, y } => {
            handler.mouse_button_down_event(btn, x, y);
        }
        DeferredEvent::MouseUp { btn, x, y } => {
            handler.mouse_button_up_event(btn, x, y);
        }
        DeferredEvent::MouseWheel { dx, dy } => {
            handler.mouse_wheel_event(dx, dy);
        }
        DeferredEvent::KeyDown { key, mods, repeat } => {
            handler.key_down_event(key, mods, repeat);
        }
        DeferredEvent::KeyUp { key, mods } => {
            handler.key_up_event(key, mods);
        }
        DeferredEvent::CharEvent { character, mods, repeat } => {
            handler.char_event(character, mods, repeat);
        }
        DeferredEvent::Resize { width, height } => {
            handler.resize_event(width, height);
        }
        DeferredEvent::Focus { has_focus } => {
            if has_focus {
                handler.window_restored_event();
            } else {
                handler.window_minimized_event();
            }
        }
        DeferredEvent::FilesDropped => {
            handler.files_dropped_event();
        }
    }
}

static mut CURSOR_ICON: crate::CursorIcon = crate::CursorIcon::Default;
static mut CURSOR_SHOW: bool = true;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sapp_touchpoint {
    pub identifier: usize,
    pub pos_x: f32,
    pub pos_y: f32,
    pub changed: bool,
}

pub fn run<F>(conf: &crate::conf::Conf, f: F)
where
    F: 'static + FnOnce() -> Box<dyn EventHandler>,
{
    {
        use std::ffi::CString;
        use std::panic;

        panic::set_hook(Box::new(|info| {
            let msg = CString::new(format!("{:?}", info)).unwrap_or_else(|_| {
                CString::new(format!("MALFORMED ERROR MESSAGE {:?}", info.location())).unwrap()
            });
            unsafe { console_log(msg.as_ptr()) };
        }));
    }

    let version = match conf.platform.webgl_version {
        crate::conf::WebGLVersion::WebGL1 => 1,
        crate::conf::WebGLVersion::WebGL2 => 2,
    };
    unsafe {
        init_webgl(version);
    }

    // setup initial canvas size
    unsafe {
        setup_canvas_size(conf.high_dpi);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    REQUESTS.with(|r| *r.borrow_mut() = Some(rx));
    let w = unsafe { canvas_width() as _ };
    let h = unsafe { canvas_height() as _ };
    let dpi_scale = unsafe { dpi_scale() };
    let clipboard = Box::new(Clipboard);
    crate::set_display(NativeDisplayData {
        blocking_event_loop: conf.platform.blocking_event_loop,
        dpi_scale,
        ..NativeDisplayData::new(w, h, tx, clipboard)
    });
    EVENT_HANDLER.with(|g| {
        *g.borrow_mut() = Some(f());
    });

    // start requestAnimationFrame loop
    unsafe {
        run_animation_loop(conf.platform.blocking_event_loop);
    }
}

pub unsafe fn sapp_width() -> ::core::ffi::c_int {
    canvas_width()
}

pub unsafe fn sapp_height() -> ::core::ffi::c_int {
    canvas_height()
}

extern "C" {
    pub fn setup_canvas_size(high_dpi: bool);
    pub fn run_animation_loop(blocking: bool);
    pub fn canvas_width() -> i32;
    pub fn canvas_height() -> i32;
    pub fn dpi_scale() -> f32;
    pub fn console_debug(msg: *const ::core::ffi::c_char);
    pub fn console_log(msg: *const ::core::ffi::c_char);
    pub fn console_info(msg: *const ::core::ffi::c_char);
    pub fn console_warn(msg: *const ::core::ffi::c_char);
    pub fn console_error(msg: *const ::core::ffi::c_char);

    pub fn sapp_set_clipboard(clipboard: *const i8, len: usize);

    /// call "requestPointerLock" and "exitPointerLock" internally.
    /// Will hide cursor and will disable mouse_move events, but instead will
    /// will make inifinite mouse field for raw_device_input event.
    /// Notice that this function will works only from "engaging" event callbacks - from
    /// "mouse_down"/"key_down" event handler functions.
    pub fn sapp_set_cursor_grab(grab: bool);

    pub fn sapp_set_cursor(cursor: *const u8, len: usize);

    pub fn sapp_is_elapsed_timer_supported() -> bool;

    pub fn sapp_set_fullscreen(fullscreen: bool);
    pub fn sapp_is_fullscreen() -> bool;
    pub fn sapp_set_window_size(new_width: u32, new_height: u32);
    pub fn sapp_schedule_update();
    pub fn init_webgl(version: i32);
    pub fn now() -> f64;
}

unsafe fn show_mouse(shown: bool) {
    if shown != CURSOR_SHOW {
        CURSOR_SHOW = shown;
        update_cursor();
    }
}

unsafe fn set_mouse_cursor(icon: crate::CursorIcon) {
    if CURSOR_ICON != icon {
        CURSOR_ICON = icon;
        if CURSOR_SHOW {
            update_cursor();
        }
    }
}

pub unsafe fn update_cursor() {
    let css_name = if !CURSOR_SHOW {
        "none"
    } else {
        match CURSOR_ICON {
            crate::CursorIcon::Default => "default",
            crate::CursorIcon::Help => "help",
            crate::CursorIcon::Pointer => "pointer",
            crate::CursorIcon::Wait => "wait",
            crate::CursorIcon::Crosshair => "crosshair",
            crate::CursorIcon::Text => "text",
            crate::CursorIcon::Move => "move",
            crate::CursorIcon::NotAllowed => "not-allowed",
            crate::CursorIcon::EWResize => "ew-resize",
            crate::CursorIcon::NSResize => "ns-resize",
            crate::CursorIcon::NESWResize => "nesw-resize",
            crate::CursorIcon::NWSEResize => "nwse-resize",
        }
    };
    sapp_set_cursor(css_name.as_ptr(), css_name.len());
}

// gl.js version required to be shipped alongside this rust code.
// "crate_version" is a misleading, but it can't be changed for legacy reasons.
#[no_mangle]
pub extern "C" fn crate_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn allocate_vec_u8(len: usize) -> *mut u8 {
    let mut string = vec![0u8; len];
    let ptr = string.as_mut_ptr();
    string.leak();
    ptr
}

static CLIPBOARD: OnceLock<Mutex<Option<String>>> = OnceLock::new();
struct Clipboard;
impl crate::native::Clipboard for Clipboard {
    fn get(&mut self) -> Option<String> {
        CLIPBOARD
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap()
            .clone()
    }

    fn set(&mut self, data: &str) {
        let len = data.len();
        let data = std::ffi::CString::new(data).unwrap();
        unsafe { sapp_set_clipboard(data.as_ptr(), len) };
    }
}

#[no_mangle]
pub extern "C" fn on_clipboard_paste(msg: *mut u8, len: usize) {
    let msg = unsafe { String::from_raw_parts(msg, len, len) };

    *CLIPBOARD.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(msg);
}

#[no_mangle]
pub extern "C" fn frame() {
    // Process any events that were deferred due to RefCell borrow conflicts
    // (e.g., touch events that fired during the previous frame's update/draw)
    drain_deferred_events();

    REQUESTS.with(|r| {
        while let Ok(request) = r.borrow_mut().as_mut().unwrap().try_recv() {
            match request {
                Request::SetCursorGrab(grab) => unsafe { sapp_set_cursor_grab(grab) },
                Request::ShowMouse(show) => unsafe { show_mouse(show) },
                Request::SetMouseCursor(cursor) => unsafe {
                    set_mouse_cursor(cursor);
                },
                Request::SetFullscreen(fullscreen) => unsafe {
                    sapp_set_fullscreen(fullscreen);
                },
                _ => {}
            }
        }
    });
    tl_event_handler(|event_handler| {
        event_handler.update();
        event_handler.draw();
    });
}

#[no_mangle]
pub extern "C" fn mouse_move(x: i32, y: i32) {
    let (x, y) = (x as f32, y as f32);
    try_event_handler_or_defer(
        DeferredEvent::MouseMove { x, y },
        |handler| handler.mouse_motion_event(x, y),
    );
}

#[no_mangle]
pub extern "C" fn raw_mouse_move(dx: i32, dy: i32) {
    let (dx, dy) = (dx as f32, dy as f32);
    try_event_handler_or_defer(
        DeferredEvent::RawMouseMove { dx, dy },
        |handler| handler.raw_mouse_motion(dx, dy),
    );
}

#[no_mangle]
pub extern "C" fn mouse_down(x: i32, y: i32, btn: i32) {
    let btn = keycodes::translate_mouse_button(btn);
    let (x, y) = (x as f32, y as f32);
    try_event_handler_or_defer(
        DeferredEvent::MouseDown { btn, x, y },
        |handler| handler.mouse_button_down_event(btn, x, y),
    );
}

#[no_mangle]
pub extern "C" fn mouse_up(x: i32, y: i32, btn: i32) {
    let btn = keycodes::translate_mouse_button(btn);
    let (x, y) = (x as f32, y as f32);
    try_event_handler_or_defer(
        DeferredEvent::MouseUp { btn, x, y },
        |handler| handler.mouse_button_up_event(btn, x, y),
    );
}

#[no_mangle]
pub extern "C" fn mouse_wheel(dx: i32, dy: i32) {
    let (dx, dy) = (dx as f32, dy as f32);
    try_event_handler_or_defer(
        DeferredEvent::MouseWheel { dx, dy },
        |handler| handler.mouse_wheel_event(dx, dy),
    );
}

#[no_mangle]
pub extern "C" fn key_down(key: u32, modifiers: u32, repeat: bool) {
    let key = keycodes::translate_keycode(key as _);
    let mods = keycodes::translate_mod(modifiers as _);
    try_event_handler_or_defer(
        DeferredEvent::KeyDown { key, mods, repeat },
        |handler| handler.key_down_event(key, mods, repeat),
    );
}

#[no_mangle]
pub extern "C" fn key_press(key: u32) {
    if let Some(character) = char::from_u32(key) {
        let mods = crate::KeyMods::default();
        try_event_handler_or_defer(
            DeferredEvent::CharEvent { character, mods, repeat: false },
            |handler| handler.char_event(character, mods, false),
        );
    }
}

#[no_mangle]
pub extern "C" fn key_up(key: u32, modifiers: u32) {
    let key = keycodes::translate_keycode(key as _);
    let mods = keycodes::translate_mod(modifiers as _);
    try_event_handler_or_defer(
        DeferredEvent::KeyUp { key, mods },
        |handler| handler.key_up_event(key, mods),
    );
}

#[no_mangle]
pub extern "C" fn resize(width: i32, height: i32) {
    // Update display dimensions immediately (not deferred)
    {
        let mut d = crate::native_display().lock().unwrap();
        d.screen_width = width as _;
        d.screen_height = height as _;
    }
    let (width, height) = (width as f32, height as f32);
    try_event_handler_or_defer(
        DeferredEvent::Resize { width, height },
        |handler| handler.resize_event(width, height),
    );
}

#[no_mangle]
pub extern "C" fn touch(phase: u32, id: u32, x: f32, y: f32) {
    let phase = keycodes::translate_touch_phase(phase as _);
    let id = id as u64;
    try_event_handler_or_defer(
        DeferredEvent::Touch { phase, id, x, y },
        |handler| handler.touch_event(phase, id, x, y),
    );
}

#[no_mangle]
pub extern "C" fn focus(has_focus: bool) {
    try_event_handler_or_defer(
        DeferredEvent::Focus { has_focus },
        |handler| {
            if has_focus {
                handler.window_restored_event();
            } else {
                handler.window_minimized_event();
            }
        },
    );
}

#[no_mangle]
pub extern "C" fn on_files_dropped_start() {
    let mut d = crate::native_display().lock().unwrap();
    d.dropped_files = Default::default();
}

#[no_mangle]
pub extern "C" fn on_files_dropped_finish() {
    try_event_handler_or_defer(
        DeferredEvent::FilesDropped,
        |handler| handler.files_dropped_event(),
    );
}

#[no_mangle]
pub extern "C" fn on_file_dropped(
    path: *mut u8,
    path_len: usize,
    bytes: *mut u8,
    bytes_len: usize,
) {
    let mut d = crate::native_display().lock().unwrap();
    let path = PathBuf::from(unsafe { String::from_raw_parts(path, path_len, path_len) });
    let bytes = unsafe { Vec::from_raw_parts(bytes, bytes_len, bytes_len) };

    d.dropped_files.paths.push(path);
    d.dropped_files.bytes.push(bytes);
}
