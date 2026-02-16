//! GUI-independent player implementation.
//!
//! It contains the player itself [`Player`] and its controller [`PlayerController`].
//!
//! [`Player`] is responsible for rendering, and can be moved to the audio thread to fill the audio
//! buffers with samples.
//!
//! [`PlayerController`] is supposed to be shared with another thread (i.e. GUI) to control the
//! player from.
//!
//! The API is straightforward. You just call [`Player::new`], which initializes a [`Player`] and
//! [`PlayerController`]. You should run [`Player::render`] within the audio loop and you can
//! control the player using the initialized controller.

use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime};

use atomic_float::AtomicF32;
use bon::Builder;
use nodi::{
    midly::{Format, MidiMessage, Smf, Timing},
    timers::TimeFormatError,
    Event as NodiEvent, MidiEvent, Moment, Sheet,
};
use ringbuf::{traits::*, HeapCons, HeapProd, HeapRb};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

/// The player engine. This type is responsible for rendering and the playback.
pub struct Player {
    sheet_receiver: HeapCons<Option<Arc<MidiSheet>>>,
    tempo_rate: Arc<AtomicF32>,
    volume: Arc<AtomicF32>,
    note_off_all_listener: HeapCons<bool>,
    is_playing: Arc<AtomicBool>,
    position: Arc<AtomicUsize>,
    previous_position: usize,
    settings: Settings,
    sheet: Option<Arc<MidiSheet>>,
    synthesizer: Synthesizer,
    tick_clock: u32, // clock in samples within a single tick
}

impl Player {
    /// Returns a tuple with `Self` and `PlayerController`.
    pub fn new(
        soundfont: &str,
        settings: Settings,
    ) -> Result<(Self, PlayerController), Box<dyn Error>> {
        let sf_file = fs::read(soundfont)?;
        let sf = SoundFont::new(&mut sf_file.as_slice())?;
        let sf = Arc::new(sf);
        let synthesizer = Synthesizer::new(&sf, &settings.clone().into())?;
        let rb = HeapRb::new(1);
        let (sheet_sender, sheet_receiver) = rb.split();
        let rb = HeapRb::new(1);
        let (note_off_all_sender, note_off_all_listener) = rb.split();
        let is_playing = Arc::new(AtomicBool::new(false));
        let position = Arc::new(AtomicUsize::new(0));
        let sample_rate = settings.sample_rate;
        let tempo_rate = Arc::new(AtomicF32::new(1.0));
        let volume = Arc::new(AtomicF32::new(1.0));

        Ok((
            Self {
                is_playing: is_playing.clone(),
                position: position.clone(),
                tempo_rate: tempo_rate.clone(),
                volume: volume.clone(),
                sheet_receiver,
                note_off_all_listener,
                settings,
                synthesizer,
                sheet: None,
                tick_clock: 0,
                previous_position: 0,
            },
            PlayerController {
                is_playing,
                position,
                tempo_rate,
                volume,
                sheet_length: Default::default(),
                sheet: None,
                sheet_sender,
                note_off_all_sender,
                cache: Cache::new(sample_rate),
            },
        ))
    }

    /// Get the settings.
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The render function which is supposed to be used within the audio loop.
    pub fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        if left.len() != right.len() {
            panic!("left and right channel buffer size cannot be different");
        }

        if let Some(should_note_off) = self.note_off_all_listener.try_pop() {
            if should_note_off {
                self.synthesizer.note_off_all(false);
                self.synthesizer.render(left, right);
                return;
            }
        }

        if !self.is_playing.load(Ordering::Relaxed) {
            self.synthesizer.render(left, right);
            return;
        }

        if let Some(sheet) = self.sheet_receiver.try_pop() {
            self.sheet = sheet;
        }

        if let Some(sheet) = &self.sheet {
            self.synthesizer
                .set_master_volume(self.volume.load(Ordering::Relaxed));
            for _ in 0..left.len() {
                let position = self.position.load(Ordering::Relaxed);
                if position == sheet.pulses.len() {
                    self.is_playing.store(false, Ordering::Relaxed);
                    self.synthesizer.note_off_all(false);
                    return;
                }

                // in case the position has been changed by the controller we reset the clock
                if position != self.previous_position {
                    self.tick_clock = 0;
                    self.previous_position = position;
                }

                let pulse = &sheet.pulses[position];

                if self.tick_clock == 0 {
                    for event in &pulse.events {
                        self.synthesizer.process_midi_message(
                            event.channel,
                            event.command,
                            event.data1,
                            event.data2,
                        );
                    }
                }

                self.tick_clock += 1;

                let pulse_duration = (pulse.duration as f32
                    * self.tempo_rate.load(Ordering::Relaxed))
                .round() as u32;

                if self.tick_clock == pulse_duration && position < sheet.pulses.len() {
                    self.position.store(position + 1, Ordering::Relaxed);
                }
            }

            self.synthesizer.render(left, right);
        }
    }
}

/// This type allows you to control the player from one thread, while rendering it on another.
pub struct PlayerController {
    is_playing: Arc<AtomicBool>,
    position: Arc<AtomicUsize>,
    tempo_rate: Arc<AtomicF32>,
    volume: Arc<AtomicF32>,
    /// The sheet length in timeline ticks.
    sheet_length: Arc<AtomicUsize>,
    sheet: Option<Arc<MidiSheet>>,
    sheet_sender: HeapProd<Option<Arc<MidiSheet>>>,
    note_off_all_sender: HeapProd<bool>,
    cache: Cache,
}

impl PlayerController {
    /// Start the playback.
    ///
    /// Returns `true` if started or already playing; `false` otherwise.
    pub fn play(&self) -> bool {
        if self.sheet.is_none() {
            self.is_playing.store(false, Ordering::SeqCst);
            return false;
        }

        let position = self.position();

        if self.is_playing() && position < 1.0 {
            return true;
        }

        if position == 1.0 {
            self.is_playing.store(false, Ordering::Relaxed);
            return false;
        }

        self.is_playing.store(true, Ordering::Relaxed);

        true
    }

    ///
    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::SeqCst)
    }

    /// Stop the playback.
    pub fn stop(&mut self) {
        if self.is_playing() {
            self.is_playing.store(false, Ordering::SeqCst);
            self.note_off_all();
        }
    }

    /// Set the playing position in timeline ticks.
    ///
    /// Values outside the valid range are clamped to `[0, total_ticks()]`. Will take effect only if
    /// a file is opened and it is not empty.
    pub fn set_position_ticks(&self, value: u64) {
        let length = self.sheet_length.load(Ordering::Relaxed);
        if length == 0 {
            return;
        }

        let tick = value.min(length as u64) as usize;
        self.position.store(tick, Ordering::Relaxed);
    }

    /// Get the current playback position in timeline ticks.
    pub fn position_ticks(&self) -> u64 {
        self.position.load(Ordering::Relaxed) as u64
    }

    /// Get the total number of timeline ticks in the loaded file.
    pub fn total_ticks(&self) -> u64 {
        self.sheet_length.load(Ordering::Relaxed) as u64
    }

    /// Set the playing position in normalized range `[0.0, 1.0]`.
    ///
    /// Will take effect only if a file is opened and it is not empty.
    pub fn set_position(&self, value: f64) {
        let total_ticks = self.total_ticks();
        if total_ticks == 0 {
            return;
        }

        let position = value.max(0.0).min(1.0);
        let tick = (total_ticks as f64 * position) as u64;
        self.set_position_ticks(tick);
    }

    /// Get normalized playing position (i.e. in range [0.0, 1.0]).
    pub fn position(&self) -> f64 {
        let total_ticks = self.total_ticks();
        if total_ticks == 0 {
            return 0.0;
        }

        let position = self.position_ticks() as f64;
        (position / total_ticks as f64).max(0.0).min(1.0)
    }

    /// Initialize a new [`PositionObserver`].
    pub fn new_position_observer(&self) -> PositionObserver {
        PositionObserver {
            position: self.position.clone(),
            length: self.sheet_length.clone(),
        }
    }

    /// Set the tempo (in beats per minute).
    pub fn set_tempo(&mut self, tempo: f32) {
        if let Some(sheet) = &mut self.sheet {
            self.tempo_rate
                .store(sheet.tempo / tempo, Ordering::Relaxed);
        }
    }

    /// Get the tempo (in beats per minute).
    ///
    /// Returns `None` if file is not set.
    pub fn tempo(&self) -> Option<f32> {
        self.sheet
            .as_ref()
            .map(|s| s.tempo / self.tempo_rate.load(Ordering::SeqCst))
    }

    /// Set master volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume.store(volume.max(0.0), Ordering::Relaxed);
    }

    /// Get master volume.
    pub fn volume(&self) -> f32 {
        self.volume.load(Ordering::Relaxed)
    }

    /// Get file duration.
    pub fn duration(&self) -> Duration {
        self.sheet
            .as_ref()
            .map(|s| s.duration())
            .unwrap_or_default()
    }

    /// Set a MIDI file.
    ///
    /// The parameter is `Option<&str>`, where `Some` value is actual path and `None` is for
    /// offloading.
    pub fn set_file(&mut self, path: Option<impl Into<PathBuf>>) -> Result<(), Box<dyn Error>> {
        match path {
            Some(path) => self.open_file(path),
            None => {
                self.offload_file();
                Ok(())
            }
        }
    }

    fn offload_file(&mut self) {
        self.stop();
        self.sheet_length.store(0, Ordering::SeqCst);
        self.sheet_sender
            .try_push(None)
            .expect("ringbuf producer must be big enough to handle new files");
        self.sheet = None;
        self.tempo_rate.store(1.0, Ordering::Relaxed);
        self.set_position(0.0);
    }

    fn open_file(&mut self, path: impl Into<PathBuf>) -> Result<(), Box<dyn Error>> {
        self.stop();
        let sheet = self.cache.open(&path.into())?;
        self.sheet_length.store(sheet.total_ticks, Ordering::SeqCst);
        self.sheet_sender
            .try_push(Some(sheet.clone()))
            .expect("ringbuf producer must be big enough to handle new files");
        self.sheet = Some(sheet);
        self.tempo_rate.store(1.0, Ordering::Relaxed);
        self.set_position(0.0);

        Ok(())
    }

    /// Send note off message for all notes (aka Panic)
    pub fn note_off_all(&mut self) {
        self.note_off_all_sender
            .try_push(true)
            .expect("ringbuf must be big enough for sending note off all message");
    }
}

struct Cache {
    sample_rate: u32,
    map: HashMap<PathBuf, Arc<MidiSheet>>,
}

impl Cache {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            map: HashMap::new(),
        }
    }

    fn open(&mut self, path: &PathBuf) -> Result<Arc<MidiSheet>, Box<dyn Error>> {
        let file = File::open(path)?;

        match self.map.get(path) {
            Some(s) => {
                if file.metadata()?.modified()? == s.modified {
                    Ok(s.clone())
                } else {
                    self.upsert(path, file)
                }
            }

            None => self.upsert(path, file),
        }
    }

    fn upsert(&mut self, path: &PathBuf, file: File) -> Result<Arc<MidiSheet>, Box<dyn Error>> {
        let sheet = Arc::new(MidiSheet::new(file, self.sample_rate)?);
        self.map.insert(path.clone(), sheet.clone());

        Ok(sheet)
    }
}

/// This type can be used to watch the playback position and update the GUI.
#[derive(Debug, Clone)]
pub struct PositionObserver {
    position: Arc<AtomicUsize>,
    length: Arc<AtomicUsize>,
}

impl PositionObserver {
    /// Get the normalized playback position in range `[0.0, 1.0]`.
    pub fn get(&self) -> f32 {
        let length = self.length.load(Ordering::Relaxed);
        if length == 0 {
            return 0.0;
        }

        self.position.load(Ordering::Relaxed) as f32 / length as f32
    }

    /// Get the current playback position in timeline ticks.
    pub fn ticks(&self) -> u64 {
        self.position.load(Ordering::Relaxed) as u64
    }

    /// Get the total number of timeline ticks in the loaded file.
    pub fn total_ticks(&self) -> u64 {
        self.length.load(Ordering::Relaxed) as u64
    }
}

/// The player settings.
#[allow(missing_docs)]
#[derive(Builder, Clone, Debug)]
pub struct Settings {
    #[builder(default = 44100)]
    pub sample_rate: u32,
    #[builder(default = 64)]
    pub block_size: u32,
    #[builder(default = 512)]
    pub audio_buffer_size: u32,
    #[builder(default = 64)]
    pub max_polyphony: u8,
    #[builder(default = true)]
    pub enable_effects: bool,
}

impl From<Settings> for SynthesizerSettings {
    fn from(settings: Settings) -> Self {
        // SynthesizerSettings is a non-exhaustive struct, so struct expressions not allowed
        let mut result = SynthesizerSettings::new(settings.sample_rate as i32);
        result.block_size = settings.block_size as usize;
        result.maximum_polyphony = settings.max_polyphony as usize;
        result.enable_reverb_and_chorus = settings.enable_effects;

        result
    }
}

#[derive(Debug, Clone)]
struct MidiSheet {
    sample_rate: u32,
    tempo: f32,
    // One pulse per MIDI tick (nodi::Moment).
    pulses: Vec<Pulse>,
    total_ticks: usize,
    modified: SystemTime,
}

impl MidiSheet {
    fn new(mut file: File, sample_rate: u32) -> Result<Self, Box<dyn Error>> {
        let modified = file.metadata()?.modified()?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let Smf { header, tracks } = Smf::parse(&buf)?;
        let ppqn = match header.timing {
            Timing::Metrical(n) => u16::from(n),
            _ => return Err(TimeFormatError.into()),
        };

        let sheet = match header.format {
            Format::SingleTrack | Format::Sequential => Sheet::sequential(&tracks),
            Format::Parallel => Sheet::parallel(&tracks),
        };
        let total_ticks = sheet.len();

        let mut duration = Pulse::duration_in_samples(500_000, ppqn as u64, sample_rate as u64);
        let tempo = sheet
            .iter()
            .flat_map(|m| &m.events)
            .find(|v| matches!(v, NodiEvent::Tempo(_)))
            .map(|v| match v {
                NodiEvent::Tempo(tempo) => us_per_beat_to_bpm(*tempo),
                _ => unreachable!(),
            })
            .unwrap_or(120.0f32);

        let pulses: Vec<Pulse> = sheet
            .iter()
            .map(|moment| Pulse::from_moment(moment, &mut duration, ppqn, sample_rate))
            .collect();
        debug_assert_eq!(pulses.len(), total_ticks);

        Ok(Self {
            sample_rate,
            pulses,
            total_ticks,
            tempo,
            modified,
        })
    }

    fn duration(&self) -> Duration {
        let duration: u64 = self.pulses.iter().map(|p| p.duration as u64).sum();
        let duration = (duration as f64 / self.sample_rate as f64) * 1_000_000.0;

        Duration::from_micros(duration as u64)
    }
}

fn us_per_beat_to_bpm(uspb: u32) -> f32 {
    60.0 / uspb as f32 * 1_000_000.0
}

#[derive(Debug, Clone)]
struct Pulse {
    // duration is in samples
    duration: u32,
    events: Vec<RawMidiEvent>,
}

impl Pulse {
    // if moment contains tempo change, new duration is calculated and set, otherwise
    // `initial_duration` is set as duration
    fn from_moment(moment: &Moment, duration: &mut u32, ppqn: u16, sample_rate: u32) -> Self {
        moment.events.iter().fold(
            Pulse {
                // we define default tempo to 120 BPM (or 500_000 us per beat)
                duration: *duration,
                events: vec![],
            },
            |mut result, event| {
                match event {
                    NodiEvent::Midi(event) => result.events.push((*event).into()),
                    NodiEvent::Tempo(tempo) => {
                        *duration = Self::duration_in_samples(
                            *tempo as u64,
                            ppqn as u64,
                            sample_rate as u64,
                        );
                        result.duration = *duration;
                    }
                    _ => (),
                }
                result
            },
        )
    }

    fn duration_in_samples(tempo_us: u64, ppqn: u64, sample_rate: u64) -> u32 {
        let numerator = (tempo_us * sample_rate) as f64;
        let denominator = (ppqn * 1_000_000) as f64;
        (numerator / denominator).round() as u32
    }
}

#[derive(Debug, Clone, Copy)]
struct RawMidiEvent {
    channel: i32, // it's i32 for compatibility with rustysynth
    command: i32,
    data1: i32,
    data2: i32,
}

impl From<MidiEvent> for RawMidiEvent {
    fn from(event: MidiEvent) -> Self {
        let channel = event.channel.as_int() as i32;

        let (command, data1, data2) = match event.message {
            MidiMessage::NoteOn { key, vel } => (0x90, key.as_int() as i32, vel.as_int() as i32),
            MidiMessage::NoteOff { key, vel } => (0x80, key.as_int() as i32, vel.as_int() as i32),
            MidiMessage::Aftertouch { key, vel } => {
                (0xA0, key.as_int() as i32, vel.as_int() as i32)
            }
            MidiMessage::Controller { controller, value } => {
                (0xB0, controller.as_int() as i32, value.as_int() as i32)
            }
            MidiMessage::ProgramChange { program } => (0xC0, program.as_int() as i32, 0),
            MidiMessage::ChannelAftertouch { vel } => (0xD0, vel.as_int() as i32, 0),
            MidiMessage::PitchBend { bend } => {
                // Adjust the bend value from [-8192, +8191] to [0, 16383]
                let midi_value = bend.as_int() as i32 + 8192;

                // Extract LSB and MSB data bytes
                let lsb = midi_value & 0x7F;
                let msb = (midi_value >> 7) & 0x7F;

                (0xE0, lsb, msb)
            }
        };

        Self {
            channel,
            command,
            data1,
            data2,
        }
    }
}
