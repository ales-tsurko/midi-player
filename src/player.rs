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

use std::collections::{BTreeSet, HashMap};
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
    midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind},
    timers::TimeFormatError,
};
use ringbuf::{traits::*, HeapCons, HeapProd, HeapRb};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

/// The player engine. This type is responsible for rendering and the playback.
pub struct Player {
    sheet_receiver: HeapCons<Option<Arc<MidiSheet>>>,
    track_filter_listener: HeapCons<TrackFilter>,
    tempo_rate: Arc<AtomicF32>,
    volume: Arc<AtomicF32>,
    note_off_all_listener: HeapCons<bool>,
    is_playing: Arc<AtomicBool>,
    position: Arc<AtomicUsize>,
    previous_position: usize,
    settings: Settings,
    sheet: Option<Arc<MidiSheet>>,
    track_filter: TrackFilter,
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
        let rb = HeapRb::new(64);
        let (track_filter_sender, track_filter_listener) = rb.split();
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
                track_filter_listener,
                note_off_all_listener,
                settings,
                synthesizer,
                sheet: None,
                track_filter: TrackFilter::new(0),
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
                track_filter_sender,
                track_filter: TrackFilter::new(0),
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
        if let Some(track_filter) = self.track_filter_listener.try_pop() {
            self.track_filter = track_filter;
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
                        if !self.track_filter.allows(event.track_index) {
                            continue;
                        }
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
    track_filter_sender: HeapProd<TrackFilter>,
    track_filter: TrackFilter,
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

    /// Returns `true` when playback is active.
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

    /// Returns metadata for all tracks in the loaded file.
    pub fn track_infos(&self) -> Vec<MidiTrackInfo> {
        self.sheet
            .as_ref()
            .map(|sheet| sheet.track_infos.clone())
            .unwrap_or_default()
    }

    /// Returns the number of tracks in the loaded file.
    pub fn track_count(&self) -> usize {
        self.sheet
            .as_ref()
            .map(|sheet| sheet.track_infos.len())
            .unwrap_or(0)
    }

    /// Mute or unmute a track by its index.
    ///
    /// Returns `true` if the state changed.
    pub fn set_track_muted(&mut self, track_index: usize, muted: bool) -> bool {
        let changed = self.track_filter.set_muted(track_index, muted);
        if changed {
            self.publish_track_filter();
            self.note_off_all();
        }
        changed
    }

    /// Returns whether a track is muted.
    ///
    /// Returns `None` if `track_index` is out of bounds.
    pub fn is_track_muted(&self, track_index: usize) -> Option<bool> {
        self.track_filter.is_muted(track_index)
    }

    /// Clears mute state for all tracks.
    ///
    /// Returns `true` if any state changed.
    pub fn clear_track_mutes(&mut self) -> bool {
        let changed = self.track_filter.clear_mutes();
        if changed {
            self.publish_track_filter();
            self.note_off_all();
        }
        changed
    }

    /// Solo or unsolo a track by its index.
    ///
    /// Returns `true` if the state changed.
    pub fn set_track_solo(&mut self, track_index: usize, soloed: bool) -> bool {
        let changed = self.track_filter.set_solo(track_index, soloed);
        if changed {
            self.publish_track_filter();
            self.note_off_all();
        }
        changed
    }

    /// Returns whether a track is soloed.
    ///
    /// Returns `None` if `track_index` is out of bounds.
    pub fn is_track_solo(&self, track_index: usize) -> Option<bool> {
        self.track_filter.is_solo(track_index)
    }

    /// Clears solo state for all tracks.
    ///
    /// Returns `true` if any state changed.
    pub fn clear_track_solos(&mut self) -> bool {
        let changed = self.track_filter.clear_solos();
        if changed {
            self.publish_track_filter();
            self.note_off_all();
        }
        changed
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
        self.track_filter = TrackFilter::new(0);
        self.publish_track_filter();
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
        self.track_filter = TrackFilter::new(self.track_count());
        self.publish_track_filter();
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

    fn publish_track_filter(&mut self) {
        let _ = self.track_filter_sender.try_push(self.track_filter.clone());
    }
}

#[derive(Debug, Clone)]
struct TrackFilter {
    muted: Vec<bool>,
    soloed: Vec<bool>,
    solo_count: usize,
}

impl TrackFilter {
    fn new(track_count: usize) -> Self {
        Self {
            muted: vec![false; track_count],
            soloed: vec![false; track_count],
            solo_count: 0,
        }
    }

    fn set_muted(&mut self, track_index: usize, muted: bool) -> bool {
        let Some(current) = self.muted.get_mut(track_index) else {
            return false;
        };
        if *current == muted {
            return false;
        }

        *current = muted;
        true
    }

    fn is_muted(&self, track_index: usize) -> Option<bool> {
        self.muted.get(track_index).copied()
    }

    fn clear_mutes(&mut self) -> bool {
        let mut changed = false;
        for muted in &mut self.muted {
            if *muted {
                *muted = false;
                changed = true;
            }
        }
        changed
    }

    fn set_solo(&mut self, track_index: usize, soloed: bool) -> bool {
        let Some(current) = self.soloed.get_mut(track_index) else {
            return false;
        };
        if *current == soloed {
            return false;
        }

        *current = soloed;
        if soloed {
            self.solo_count += 1;
        } else {
            self.solo_count = self.solo_count.saturating_sub(1);
        }
        true
    }

    fn is_solo(&self, track_index: usize) -> Option<bool> {
        self.soloed.get(track_index).copied()
    }

    fn clear_solos(&mut self) -> bool {
        if self.solo_count == 0 {
            return false;
        }

        for soloed in &mut self.soloed {
            *soloed = false;
        }
        self.solo_count = 0;
        true
    }

    fn allows(&self, track_index: usize) -> bool {
        let muted = self.muted.get(track_index).copied().unwrap_or(false);
        let soloed = self.soloed.get(track_index).copied().unwrap_or(false);
        if self.solo_count > 0 {
            soloed && !muted
        } else {
            !muted
        }
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

/// MIDI track metadata extracted from the loaded file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidiTrackInfo {
    /// Zero-based track index.
    pub index: usize,
    /// Optional track name from MIDI meta events.
    pub name: Option<String>,
    /// Distinct MIDI channels used by this track (1-based, range `1..=16`).
    pub channels: Vec<u8>,
    /// Distinct MIDI program numbers used by this track.
    pub programs: Vec<u8>,
}

#[derive(Debug, Clone)]
struct MidiSheet {
    sample_rate: u32,
    tempo: f32,
    // One pulse per MIDI tick (nodi::Moment).
    pulses: Vec<Pulse>,
    total_ticks: usize,
    track_infos: Vec<MidiTrackInfo>,
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
        let mut tick_events: HashMap<u64, Vec<RawMidiEvent>> = HashMap::new();
        let mut tempo_changes: HashMap<u64, u32> = HashMap::new();
        let mut max_tick = 0_u64;
        let mut track_infos = Vec::with_capacity(tracks.len());

        for (track_index, track) in tracks.iter().enumerate() {
            let mut absolute_tick = 0_u64;
            let mut name = None;
            let mut channels = BTreeSet::new();
            let mut programs = BTreeSet::new();

            for event in track {
                absolute_tick = absolute_tick.saturating_add(u64::from(event.delta.as_int()));
                max_tick = max_tick.max(absolute_tick);

                match event.kind {
                    TrackEventKind::Midi { channel, message } => {
                        channels.insert(channel.as_int() + 1);
                        if let MidiMessage::ProgramChange { program } = message {
                            programs.insert(program.as_int());
                        }
                        tick_events.entry(absolute_tick).or_default().push(
                            RawMidiEvent::from_track_event(track_index, channel.as_int(), message),
                        );
                    }
                    TrackEventKind::Meta(MetaMessage::Tempo(tempo)) => {
                        tempo_changes.insert(absolute_tick, tempo.as_int());
                    }
                    TrackEventKind::Meta(MetaMessage::TrackName(track_name)) if name.is_none() => {
                        let decoded = String::from_utf8_lossy(track_name).trim().to_string();
                        if !decoded.is_empty() {
                            name = Some(decoded);
                        }
                    }
                    _ => {}
                }
            }

            track_infos.push(MidiTrackInfo {
                index: track_index,
                name,
                channels: channels.into_iter().collect(),
                programs: programs.into_iter().collect(),
            });
        }

        let total_ticks_u64 = if tracks.is_empty() {
            0
        } else {
            max_tick.saturating_add(1)
        };
        let total_ticks = usize::try_from(total_ticks_u64)
            .map_err(|_| "MIDI timeline is too large for this platform")?;

        let initial_tempo = tempo_changes
            .iter()
            .min_by_key(|(tick, _tempo)| **tick)
            .map(|(_tick, tempo)| *tempo)
            .unwrap_or(500_000);
        let tempo = us_per_beat_to_bpm(initial_tempo);
        let mut duration =
            Pulse::duration_in_samples(u64::from(initial_tempo), ppqn as u64, sample_rate as u64);
        let mut pulses = Vec::with_capacity(total_ticks);

        for tick in 0..total_ticks {
            if let Some(tempo) = tempo_changes.get(&(tick as u64)) {
                duration =
                    Pulse::duration_in_samples(u64::from(*tempo), ppqn as u64, sample_rate as u64);
            }

            pulses.push(Pulse {
                duration,
                events: tick_events.remove(&(tick as u64)).unwrap_or_default(),
            });
        }

        Ok(Self {
            sample_rate,
            pulses,
            total_ticks,
            tempo,
            track_infos,
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
    fn duration_in_samples(tempo_us: u64, ppqn: u64, sample_rate: u64) -> u32 {
        let numerator = (tempo_us * sample_rate) as f64;
        let denominator = (ppqn * 1_000_000) as f64;
        (numerator / denominator).round() as u32
    }
}

#[derive(Debug, Clone, Copy)]
struct RawMidiEvent {
    track_index: usize,
    channel: i32, // it's i32 for compatibility with rustysynth
    command: i32,
    data1: i32,
    data2: i32,
}

impl RawMidiEvent {
    fn from_track_event(track_index: usize, channel: u8, message: MidiMessage) -> Self {
        let channel = i32::from(channel);

        let (command, data1, data2) = match message {
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
            track_index,
            channel,
            command,
            data1,
            data2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrackFilter;

    #[test]
    fn mute_and_solo_rules_follow_daw_semantics() {
        let mut filter = TrackFilter::new(3);

        assert!(filter.allows(0));
        assert!(filter.allows(1));
        assert!(filter.allows(2));

        assert!(filter.set_muted(1, true));
        assert!(filter.allows(0));
        assert!(!filter.allows(1));
        assert!(filter.allows(2));

        assert!(filter.set_solo(2, true));
        assert!(!filter.allows(0));
        assert!(!filter.allows(1));
        assert!(filter.allows(2));

        assert!(filter.set_muted(2, true));
        assert!(!filter.allows(2));

        assert!(filter.clear_solos());
        assert!(filter.allows(0));
        assert!(!filter.allows(1));
        assert!(!filter.allows(2));
    }

    #[test]
    fn out_of_bounds_track_changes_are_ignored() {
        let mut filter = TrackFilter::new(2);

        assert!(!filter.set_muted(4, true));
        assert!(!filter.set_solo(4, true));
        assert_eq!(filter.is_muted(4), None);
        assert_eq!(filter.is_solo(4), None);
    }
}
