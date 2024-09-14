#!/usr/bin/env python3
# Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# ... [License text] ...

"""
Convert wake word model from Keras to TensorFlow

:model str
    Input Keras model (.net)

    :-o --out str {model}.tflite
        Custom output TensorFlow Lite filename. Must be a folder name if output format is 'saved_model'.

    :-f --format str {model}.tflite
        Custom output TensorFlow Lite filename
"""
import os
from os.path import split, isfile, splitext
from prettyparse import Usage
from pathlib import Path

from precise.scripts.base_script import BaseScript

class ConvertScript(BaseScript):
    usage = Usage(__doc__)

    def run(self):
        args = self.args
        input_path = Path(args.model)
        input_model_filename = input_path.name
        model_name = input_model_filename.replace('.net', '')
        output_format = args.format.lower()
        out_file = args.out.format(model=model_name, ext=self.get_extension(output_format))
        self.convert(args.model, input_model_filename, out_file, output_format)

    def get_extension(self, output_format: str) -> str:
        if output_format == 'tflite' or output_format == 'trainable_tflite':
            return 'tflite'
        elif output_format == 'saved_model':
            return ''
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def convert(self, model_path: str, model_name:str, out_file: str, output_format: str):
        """
        Converts a Keras model to the specified TensorFlow format.

        Args:
            model_path: Path to the Keras model (.h5 or .net)
            out_file: Path to save the converted model
            output_format: The desired output format ('tflite', 'saved_model', 'trainable_tflite')
        """
        print(f'Converting {model_path} to {out_file} as {output_format}...')

        import tensorflow as tf  # Using TensorFlow v2.8
        from tensorflow import keras as K
        from precise.functions import weighted_log_loss

        out_dir, filename = split(out_file)
        out_dir = out_dir or '.'
        os.makedirs(out_dir, exist_ok=True)

        # Load custom loss function with model
        model = K.models.load_model(model_path, custom_objects={'weighted_log_loss': weighted_log_loss})
        model.summary()

        if output_format == 'tflite':
            # Convert to frozen TFLite model
            print('Starting TFLite conversion.')
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite's built-in operations
                tf.lite.OpsSet.SELECT_TF_OPS  # Enable full TensorFlow operations
            ]
            # Disable experimental lowering of tensor list ops
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            with open(out_file, "wb") as f:
                f.write(tflite_model)
            print('Wrote TFLite model to ' + out_file)

        elif output_format == 'saved_model':
            # Save the model as a TensorFlow SavedModel
            model.save(out_file, save_format='tf')
            print('Saved Keras model to SavedModel format at', out_file)

        elif output_format == 'trainable_tflite':
            # Save the model as a SavedModel first
            saved_model_dir = os.path.join(out_dir, 'saved_model_temp')
            model.save(saved_model_dir, save_format='tf')
            print('Saved Keras model to SavedModel format at', saved_model_dir)

            # Convert the SavedModel to a trainable TFLite model
            print('Starting TFLite conversion for trainable model.')
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

            # Enable experimental features for training
            converter.experimental_enable_resource_variables = True

            # Set supported operations
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]

            tflite_model = converter.convert()
            with open(out_file, "wb") as f:
                f.write(tflite_model)
            print('Wrote trainable TFLite model to ' + out_file)

            # Optionally, clean up the temporary SavedModel directory
            import shutil
            shutil.rmtree(saved_model_dir)

        else:
            raise ValueError(f"Unknown output format: {output_format}")

main = ConvertScript.run_main

if __name__ == '__main__':
    main()