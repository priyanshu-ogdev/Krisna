mod pipeline;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(pipeline::PipelineState::default())
        .invoke_handler(tauri::generate_handler![
            pipeline::pipeline_snapshot,
            pipeline::record_user_turn,
            pipeline::record_self_review,
            pipeline::record_finalize,
            pipeline::record_rollback,
            pipeline::begin_revision_cycle,
            pipeline::check_inactivity,
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while building tauri application");
}
