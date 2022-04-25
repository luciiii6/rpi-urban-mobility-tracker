from .validation_exception import ValidationException
import os


class Validator:
    @staticmethod
    def validate_args(args):
        print(args.placement)
        if args.model_path:
            if not args.label_map_path:
                raise ValidationException(
                    "when specifying a custom model, you must also specify a label map path using: '-labelmap <path to labelmap.txt>'")
        if args.model_path:
            if not os.path.exists(args.model_path) == True:
                raise ValidationException("can't find the specified model...")
        if args.label_map_path:
            if not os.path.exists(args.label_map_path) == True:
                raise ValidationException(
                    "can't find the specified label map...")
        if args.video_path:
            if not os.path.exists(args.video_path) == True:
                raise ValidationException(
                    "can't find the specified video file...")
        if not (args.placement == "facing" or args.placement == "above"): 
            raise ValidationException(
                """specify the placement of the camera using '-placement facing/above' meaning the camera is above the door or facing the door""")
