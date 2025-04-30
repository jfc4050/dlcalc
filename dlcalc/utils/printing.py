_BOLD = "\033[1m"
_END = "\033[0m"


def print_bold(text: str) -> None:
    print(_BOLD + text + _END)


def print_h1_header(section_name: str) -> None:
    print()
    print_bold("--------------------------------------------------------------------------")
    print_bold(section_name)
    print_bold("--------------------------------------------------------------------------")


def print_h2_header(section_name: str) -> None:
    print_bold(section_name)
    print_bold("-----------------------------------------")
