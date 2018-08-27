from flask import jsonify, request

from servicex import flaskx


app = flaskx.make_app(__name__)
app.config.from_object(__name__)
basic_auth = flaskx.make_system_auth_middleware(app)


@app.route("/greeting/<name>")
@basic_auth.required
def greeting(name):
    suffix = request.args.get("suffix", "!")
    return jsonify(dict(message=f"Hello, {name}{suffix}"))
