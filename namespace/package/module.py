class DemoClass:
    def __init__(self, name: str):
        self.name: str = name

    def demo_method(self, your_name: str) -> str:
        """Simple Hello <your_name> demo

        Args:
            your_name: your name

        Returns:
            a string containing a message to you.
        """
        return f"Hello {your_name} from the {self.__class__.__name__}."
