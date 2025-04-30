extends Control

@onready var crt_overlay: SubViewportContainer = %CRTOverlay
@onready var processed_face_viewport: SubViewport = %ProcessedFaceViewport

var url = 'http://127.0.0.1:5000/'

func _ready() -> void:
	set_random_image()

func _process(delta: float) -> void:
	if Engine.get_process_frames() % 10 == 0:
		update_shader()

func update_shader() -> void:
	crt_overlay.material.set_shader_parameter("scanlines_opacity", %ScanlinesOpacity.value)
	crt_overlay.material.set_shader_parameter("scanlines_width", %ScanlinesWidth.value)
	crt_overlay.material.set_shader_parameter("grille_opacity", %GrilleOpacity.value)
	crt_overlay.material.set_shader_parameter("roll_speed", %RollSpeed.value)
	crt_overlay.material.set_shader_parameter("roll_size", %RollSize.value)
	crt_overlay.material.set_shader_parameter("roll_variation", %RollVariation.value)
	crt_overlay.material.set_shader_parameter("distort_intensity", %DistortIntensity.value)
	crt_overlay.material.set_shader_parameter("noise_opacity", %NoiseOpacity.value)
	crt_overlay.material.set_shader_parameter("noise_speed", %NoiseSpeed.value)
	crt_overlay.material.set_shader_parameter("static_noise_intensity", %NoiseIntensity.value)
	crt_overlay.material.set_shader_parameter("aberration", %Aberration.value)
	crt_overlay.material.set_shader_parameter("brightness", %Brightness.value)
	crt_overlay.material.set_shader_parameter("warp_amount", %WarpAmount.value)
	crt_overlay.material.set_shader_parameter("vignette_intensity", %VignetteIntensity.value)
	crt_overlay.material.set_shader_parameter("vignette_opacity", %VignetteOpacity.value)
	crt_overlay.material.set_shader_parameter("vignette_opacity", %VignetteOpacity.value)
	crt_overlay.material.set_shader_parameter("resolution", Vector2(%Resolution.value, %Resolution.value))
	crt_overlay.material.set_shader_parameter("tint_value", %Tint.value)
	crt_overlay.material.set_shader_parameter("temp_value", %Temp.value)


func send_http_deepface_request(image: Image) -> void:
	print('Sending request')
	var base_64_data = Marshalls.raw_to_base64(image.save_png_to_buffer())
	var body = JSON.new().stringify({
		"image": str('data:image/png;base64,',base_64_data)
		})
	var headers: PackedStringArray = ['Content-type:application/json']
	%HTTPRequest.request(url+"predict", headers, HTTPClient.METHOD_POST, body)

func set_random_image() -> void:
	var paths: Array[String] = []
	var dir_path := "res://Assets/Sample Images/"
	var dir = DirAccess.open(dir_path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "":
			if !dir.current_is_dir() and file_name.ends_with(".png"):
				paths.append(dir_path+"/"+file_name)
			file_name = dir.get_next()
		crt_overlay.material.set_shader_parameter("face_image", load(paths[randi_range(0,len(paths)-1)]))
		#face_material.albedo_texture = load(paths[randi_range(0,len(paths)-1)])
	else:
		print("An error occurred when trying to access the path.")


func _on_check_spoof_button_button_up() -> void:
	#processed_face_viewport.get_texture().get_image().save_png("res://Assets/SavedImages/overlay.png")
	send_http_deepface_request(processed_face_viewport.get_texture().get_image())


func _on_http_request_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		printerr('HTTP Error ',response_code)
		return
	# Get the JSON response and parse it
	var json = JSON.new()
	json.parse(body.get_string_from_utf8())
	var response = json.get_data()

	print(response)


func _on_new_face_button_up() -> void:
	set_random_image()

func get_all_children(node) -> Array:
	var nodes: Array = []
	for N: Node in node.get_children():
		if N.get_child_count() > 0:
			nodes.append(N)
			nodes.append_array(get_all_children(N))
		else:
			nodes.append(N)
	return nodes
