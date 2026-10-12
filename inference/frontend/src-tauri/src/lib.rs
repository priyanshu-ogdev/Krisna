pub mod backend_client;
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
            pipeline::backend_health,
            pipeline::backend_ready,
            pipeline::backend_models_status,
            pipeline::backend_warm_models,
            pipeline::backend_warm_status,
            pipeline::backend_unload_models,
            pipeline::backend_create_session,
            pipeline::backend_get_session,
            pipeline::backend_session_model_status,
            pipeline::backend_send_message,
            pipeline::backend_trigger_self_review,
            pipeline::backend_finalize,
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
