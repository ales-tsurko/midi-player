#![allow(missing_docs)]

use midi_player::player::{Player, Settings};

fn main() {
    let settings = Settings::builder().build();
    let mut player = Player::new(
        settings,
        "examples/Nice-Steinway-Lite-v3.0.sf2",
        "examples/Sibelius_The_Spruce.mid",
    )
    .unwrap();
    player.play();
}
