@tool
extends EditorScript

func _run() -> void:
	var progress_bar_theme: Gradient = load("res://Resources/Themes/progress_bar_gradient.tres")
	print(progress_bar_theme.sample(0.1))
