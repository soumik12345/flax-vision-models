from flax_models.vgg import build_vgg16, build_vgg19


model, parameters = build_vgg19(show_parameter_overview=True)
