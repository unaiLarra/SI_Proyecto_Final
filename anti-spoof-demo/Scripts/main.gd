extends Control

var url = 'http://127.0.0.1:5000/'

func _ready() -> void:
	set_random_image()

func send_http_deepface_request(image: Image, function_name: String) -> void:
	print('Sending request with image: ',image,' , function_name: ',function_name)
	var base_64_data = Marshalls.raw_to_base64(image.save_png_to_buffer())
	var body = JSON.new().stringify({
		"image": str('data:image/png;base64,',base_64_data)
		})
	var headers: PackedStringArray = ['Content-type:application/json']
	%HTTPRequest.request(url+"predict", headers, HTTPClient.METHOD_POST, body)

func set_random_image() -> void:
	var dir_path := "res://Assets/Sample Images/"
	var dir = DirAccess.open(dir_path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "":
			if !dir.current_is_dir():
				%FaceImage.texture = load(dir_path+"/"+file_name)
			file_name = dir.get_next()
	else:
		print("An error occurred when trying to access the path.")
