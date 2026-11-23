"""兼容 python -m fetch。"""
from fetch.control.fetch import main
raise SystemExit(main())
