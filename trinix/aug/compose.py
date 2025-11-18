from typing import Any, Callable, Dict, List, Optional, Union


class Compose:
    """
    Composes several transforms together.

    Args:
        transforms: List of transform functions/objects to compose.
        p: Probability of applying the entire composition (default: 1.0).

    Example:
        >>> transforms = Compose([
        ...     RandomFlip(p=0.5),
        ...     RandomRotate(degrees=15),
        ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
        >>> augmented = transforms(image=image)
    """

    def __init__(
        self,
        transforms: List[Union[Callable, Any]],
        p: float = 1.0,
    ):
        if not isinstance(transforms, list):
            raise TypeError(f"transforms must be a list, got {type(transforms)}")

        self.transforms = transforms
        self.p = p

    def __call__(self, *args, **kwargs) -> Any:
        import random

        if random.random() > self.p:
            if args:
                return args[0] if len(args) == 1 else args
            return kwargs

        data = kwargs if kwargs else args

        for transform in self.transforms:
            if callable(transform):
                if isinstance(data, dict):
                    data = transform(**data)
                elif isinstance(data, tuple):
                    data = transform(*data)
                else:
                    data = transform(data)
            else:
                raise TypeError(f"Transform must be callable, got {type(transform)}")

        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        if self.p < 1.0:
            format_string += f", p={self.p}"
        return format_string

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int):
        return self.transforms[idx]
