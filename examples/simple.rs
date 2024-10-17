#![allow(missing_docs)]

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

    thread::spawn(move || {
        thread::sleep(Duration::from_secs(2));
        controller
            .open_file("examples/Sibelius_The_Spruce.mid")
            .unwrap();

        println!(
            "file duration in seconds: {}",
            controller.duration().as_secs()
        );

        controller.play();

        let position_observer = controller.new_position_observer();

        thread::spawn(move || loop {
            println!("playhead position is: {}", position_observer.get());
            thread::sleep(Duration::from_secs(1));
        });
    });

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
