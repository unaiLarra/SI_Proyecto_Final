extends Control

var url = 'http://127.0.0.1:5000/'

func _ready() -> void:
	set_random_image()

func _process(delta: float) -> void:
	%FaceImage.material.set_shader_parameter("scanlines_opacity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("scanlines_width", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("grille_opacity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("roll_speed", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("roll_size", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("roll_variation", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("distort_intensity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("noise_opacity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("noise_speed", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("static_noise_intensity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("aberration", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("brightness", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("warp_amount", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("vignette_intensity", %ScanlinesOpacity.value)
	%FaceImage.material.set_shader_parameter("vignette_opacity", %ScanlinesOpacity.value)


func send_http_deepface_request(image: Image, function_name: String) -> void:
	print('Sending request with image: ',image,' , function_name: ',function_name)
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
		%FaceImage.texture = load(paths[randi_range(0,len(paths))])
	else:
		print("An error occurred when trying to access the path.")
