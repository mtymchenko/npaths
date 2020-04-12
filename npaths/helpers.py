import re


__all__ = [
    'read_str_sparam'
]


def read_str_sparam(sparam):
    """Reads the string S-parameter and returns ports' and
    harmonics' indexes.

    Args:
        sparam (string): String S-parameter:

    Example:
        To return the S_{2,1} at the fundamental harmonics,
        we can write either of the following:
    >>> ``Block._read_str_sparam('S(2, 1)')`` \\
    >>> ``Block._read_str_sparam('S(2:0, 1:0)')``

    Returns:
        port_to (2-tuple of ints): destination port and harmonic.
        port_from (2-tuple of ints): source port and harmonic.

    """
    sparam_string = re.sub(r'([()])', r'\\\1', sparam)

    port_to = re.findall(r'\((.*?)\,', sparam_string)[0]
    port_from = re.findall(r'\,(.*?)\)', sparam_string)[0]

    port_to = list(map(int, re.findall(r'-?\d+', port_to)))
    if len(port_to) == 1:
        port_to.append(0)
    port_to = tuple(port_to)

    port_from = list(map(int, re.findall(r'-?\d+', port_from)))
    if len(port_from) == 1:
        port_from.append(0)
    port_from = tuple(port_from)

    return tuple([port_to, port_from])