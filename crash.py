from dinov3_jax.eepynox.debug import debugpy_pm

# @pytest.hookimpl(tryfirst=True)
# def pytest_exception_interact(call: pytest.CallInfo):
#     print(f"pytest_exception_interact called with call: {call}")
#     if call.when == 'call' and call.excinfo and call.excinfo._excinfo:
#         maybe_debugpy_postmortem(call.excinfo._excinfo)
# print(sys.monitoring.get_events(sys.monitoring.DEBUGGER_ID))
# sys.monitoring.set_events(sys.monitoring.DEBUGGER_ID,0)
with debugpy_pm():
    import time

    time.sleep(10.0)
    raise RuntimeError("This is a test crash for debugpy post-mortem debugging.")
