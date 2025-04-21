extends Control

var url = 'http://127.0.0.1:5000/'

func send_http_deepface_request(image: Image, function_name: String) -> void:
	print('Sending request with image: ',image,' , function_name: ',function_name)
	var base_64_data = Marshalls.raw_to_base64(image.save_png_to_buffer())
	var body = JSON.new().stringify({
		"image": str('data:image/png;base64,',base_64_data)
		})
	var headers: PackedStringArray = ['Content-type:application/json']
	%HTTPRequest.request(url+"predict", headers, HTTPClient.METHOD_POST, body)
