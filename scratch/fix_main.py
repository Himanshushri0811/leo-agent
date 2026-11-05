with open("app/main.py", "rb") as f:
    content = f.read()

target = b"        except Exception as exc:\r\n        logger.error(\"database_init_failed\", error=str(exc))\r\n        if settings.is_production:\r\n            raise"
replacement = b"        except Exception as exc:\r\n            logger.error(\"database_init_failed\", error=str(exc))\r\n            if settings.is_production:\r\n                raise"

if target not in content:
    target = b"        except Exception as exc:\n        logger.error(\"database_init_failed\", error=str(exc))\n        if settings.is_production:\n            raise"
    replacement = b"        except Exception as exc:\n            logger.error(\"database_init_failed\", error=str(exc))\n            if settings.is_production:\n                raise"

assert target in content, "target not found"
content = content.replace(target, replacement)
with open("app/main.py", "wb") as f:
    f.write(content)
print("SUCCESS")

