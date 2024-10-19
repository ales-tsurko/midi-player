#![allow(missing_docs)]

use std::{thread, time::Duration};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    StreamConfig,
};
use indicatif::{ProgressBar, ProgressStyle};
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

    println!("tempo: {:.2} BPM", controller.tempo().unwrap());

    let position_observer = controller.new_position_observer();

    thread::spawn(move || {
        let pb = ProgressBar::new(1000);
        pb.set_style(
            ProgressStyle::with_template("{wide_bar:.magenta.on_white/blue.on_white}")
                .unwrap()
                .progress_chars("━━-"),
        );

        loop {
            let position = (1000.0 * position_observer.get()) as u64;
            pb.set_position(position);
            thread::sleep(Duration::from_millis(100));
        }
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
