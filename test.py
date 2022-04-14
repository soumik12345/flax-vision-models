from flax_models.vgg import build_vgg16


model, parameters = build_vgg16(show_parameter_overview=True)
