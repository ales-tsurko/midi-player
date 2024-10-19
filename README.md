midi-player
===========

A MIDI file player library with integrated synthesizer ([`rustysynth`](https://crates.io/crates/rustysynth)).

It's independent from audio library and provides optional GUI written in
**iced**.

Has all the basic features: play/stop, position, volume, tempo parameters and
position observer, which allows you to track playhead position in the real-time. 

The player controller is separated from the player engine and can be used on a
different thread (i.e from GUI).




## Example usage (with `cpal`)


```rust
use std::{thread, time::Duration};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use midi_player::player::{Player, Settings};

fn main() {
    let settings = Settings::builder().build();
    let (player, mut controller) =
        Player::new("examples/Nice-Steinway-Lite-v3.0.sf2", settings).unwrap();

    thread::spawn(|| {
        start_audio_loop(player);
    });

    thread::sleep(Duration::from_secs(2));

    controller
        .set_file(Some("examples/Sibelius_The_Spruce.mid"))
        .unwrap();

    controller.play();

    loop {}
}

fn start_audio_loop(mut player: Player) {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");
    let channels = 2 as usize;
    let config = StreamConfig {
        channels: channels as u16,
        sample_rate: cpal::SampleRate(player.settings().sample_rate),
        buffer_size: cpal::BufferSize::Fixed(player.settings().audio_buffer_size),
    };

    let err_fn = |err| eprintln!("An error occurred on the output audio stream: {}", err);

    let mut left = vec![0f32; player.settings().audio_buffer_size as usize];
    let mut right = vec![0f32; player.settings().audio_buffer_size as usize];

    let stream = device
        .build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let sample_count = data.len() / channels;

                player.render(&mut left, &mut right);

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

    std::thread::park();
}
```
