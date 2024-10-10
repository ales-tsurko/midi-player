//! GUI-independent player implementation.

use std::error::Error;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

use bon::Builder;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use nodi::{
    midly::{Format, MidiMessage, Smf},
    timers::Ticker,
    Connection, MidiEvent, Player as NodiPlayer, Sheet,
};
use ringbuf::{traits::*, HeapCons, HeapProd, HeapRb};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

#[allow(missing_docs)]
pub struct Player {
    is_playing: bool,
    settings: Settings,
    player: NodiPlayer<Ticker, SynthesizerWrapper>,
    sheet: Sheet,
}

impl Player {
    #[allow(missing_docs)]
    pub fn new(settings: Settings, soundfont: &str, file: &str) -> Result<Self, Box<dyn Error>> {
        let sf_file = fs::read(soundfont)?;
        let sf = SoundFont::new(&mut sf_file.as_slice())?;
        let sf = Arc::new(sf);
        let synthesizer = SynthesizerWrapper::new(&sf, &settings)?;

        let file = fs::read(file)?;
        let Smf { header, tracks } = Smf::parse(&file)?;
        let timer = Ticker::try_from(header.timing)?;
        let sheet = match header.format {
            Format::SingleTrack | Format::Sequential => Sheet::sequential(&tracks),
            Format::Parallel => Sheet::parallel(&tracks),
        };

        let player = NodiPlayer::new(timer, synthesizer);

        Ok(Self {
            is_playing: false,
            settings,
            player,
            sheet,
        })
    }

    /// Set the MIDI file to play.
    pub fn set_file(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.stop();

        let data = fs::read(path)?;
        let Smf { header, tracks } = Smf::parse(&data)?;
        let timer = Ticker::try_from(header.timing)?;

        let sheet = match header.format {
            Format::SingleTrack | Format::Sequential => Sheet::sequential(&tracks),
            Format::Parallel => Sheet::parallel(&tracks),
        };

        self.sheet = sheet;
        self.player.set_timer(timer);

        Ok(())
    }

    /// Set the tempo.
    pub fn set_tempo(&mut self) {}

    /// Start the playback.
    pub fn play(&mut self) {
        if !self.is_playing {
            self.player.con.set_should_stop(false);
            self.is_playing = true;
            // TODO: the position should be taken into account here
            self.player.play(&self.sheet);
        }
    }

    /// Stop the playback.
    pub fn stop(&mut self) {
        if self.is_playing {
            self.player.con.set_should_stop(true);
            self.is_playing = false;
        }
    }

    /// Get current playing position ([0.0, 1.0]).
    pub fn position(&self) -> f64 {
        todo!()
    }

    /// Set playhead position in range [0.0, 1.0].
    pub fn set_position(&mut self, position: f64) {
        let was_playing = self.is_playing;

        self.stop();

        let position = position.max(0.0).min(1.0);

        if was_playing {
            self.play();
        }
    }

    /// Get the file Duration.
    pub fn file_duration(&self) -> Duration {
        todo!()
    }
}

struct SynthesizerWrapper {
    should_stop: bool,
    event_sender: HeapProd<(MidiEvent, bool)>,
    stream: cpal::Stream,
}

impl SynthesizerWrapper {
    fn new(
        soundfont: &Arc<SoundFont>,
        settings: &Settings,
    ) -> Result<Self, Box<dyn Error>> {
        let (stream, event_sender) = Self::start_renderer(soundfont, settings)?;

        Ok(Self {
            should_stop: false,
            event_sender,
            stream,
        })
    }

    fn start_renderer(
        soundfont: &Arc<SoundFont>,
        settings: &Settings,
    ) -> Result<(cpal::Stream, HeapProd<(MidiEvent, bool)>), Box<dyn Error>> {
        // the first item is the event, the second is whether to call all notes off message
        let buf = HeapRb::<(MidiEvent, bool)>::new(64);
        let (sender, mut receiver) = buf.split();
        let mut synth = Synthesizer::new(soundfont, &settings.clone().into())?;

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("No output device available");
        let channels = 2 as usize;
        let config = StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(synth.get_sample_rate() as u32),
            buffer_size: cpal::BufferSize::Fixed(settings.audio_buffer_size),
        };

        let err_fn = |err| eprintln!("An error occurred on the output audio stream: {}", err);

        let mut left = vec![0f32; settings.audio_buffer_size as usize];
        let mut right = vec![0f32; settings.audio_buffer_size as usize];

        let stream = device
            .build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let sample_count = data.len() / channels;

                    if let Some((event, should_note_off_all)) = receiver.try_pop() {
                        if should_note_off_all {
                            synth.note_off_all(false);
                        } else {
                            let (channel, command, data1, data2) = event_to_raw(event);
                            synth.process_midi_message(channel, command, data1, data2);
                        }
                    }

                    synth.render(&mut left, &mut right);

                    if !left.is_empty() {
                        for i in 0..sample_count {
                            data[channels * i] = left[i];
                            data[channels * i + 1] = right[i];
                        }
                    }
                },
                err_fn,
                None,
            )
            .unwrap();

        stream.play().expect("cannot run audio stream");

        Ok((stream, sender))
    }

    fn set_should_stop(&mut self, should_stop: bool) {
        self.should_stop = should_stop;
    }
}

impl Connection for SynthesizerWrapper {
    fn play(&mut self, event: MidiEvent) -> bool {
        self.event_sender.try_push((event, false)).is_ok() && !self.should_stop
    }

    fn all_notes_off(&mut self) {
        let _ = self.event_sender.try_push((default_midi_event(), true));
    }
}

fn default_midi_event() -> MidiEvent {
    MidiEvent {
        channel: 0.into(),
        message: MidiMessage::NoteOn {
            key: 60.into(),
            vel: 100.into(),
        },
    }
}

fn event_to_raw(event: MidiEvent) -> (i32, i32, i32, i32) {
    let channel = event.channel.as_int() as i32;

    let (command, data1, data2) = match event.message {
        MidiMessage::NoteOn { key, vel } => (0x90, key.as_int() as i32, vel.as_int() as i32),
        MidiMessage::NoteOff { key, vel } => (0x80, key.as_int() as i32, vel.as_int() as i32),
        MidiMessage::Aftertouch { key, vel } => (0xA0, key.as_int() as i32, vel.as_int() as i32),
        MidiMessage::Controller { controller, value } => {
            (0xB0, controller.as_int() as i32, value.as_int() as i32)
        }
        MidiMessage::ProgramChange { program } => (0xC0, program.as_int() as i32, 0),
        MidiMessage::ChannelAftertouch { vel } => (0xD0, vel.as_int() as i32, 0),
        MidiMessage::PitchBend { bend } => {
            // Adjust the bend value from [-8192, +8191] to [0, 16383]
            let midi_value = bend.as_int() as i32 + 8192;

            // Extract LSB and MSB data bytes
            let lsb = (midi_value & 0x7F) as i32;
            let msb = ((midi_value >> 7) & 0x7F) as i32;

            (0xE0, lsb, msb)
        }
    };

    (channel, command, data1, data2)
}

/// The player settings.
#[derive(Builder, Clone, Debug)]
pub struct Settings {
    #[builder(default = 44100)]
    sample_rate: u32,
    #[builder(default = 64)]
    block_size: u32,
    #[builder(default = 256)]
    audio_buffer_size: u32,
    #[builder(default = 64)]
    max_polyphony: u8,
    #[builder(default = true)]
    enable_effects: bool,
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
