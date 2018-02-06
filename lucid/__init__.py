""" Logging in lucid

Lucid uses Python's default logging system.
Each module has it's own logger, so you can control the verbosity of each
module to your needs. To change the overall log level, you can set log levels
at each part of the module hierarchy, including simply at the root, `lucid`:

```python
import logging
logging.getLogger('lucid').setLevel(logging.INFO)
```

Our log levels also confirm to Python's defaults:

DEBUG	Detailed information typically of interest only when diagnosing problems.
INFO	Confirmation that things are working as expected.
WARN	An indication that something unexpected happened.
ERROR	The software has not been able to perform some function.
CRITICAL	The program itself may be unable to continue running.

"""

import logging
logging.basicConfig(level=logging.WARN)
