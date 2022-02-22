from waitress import serve
import m3d_vton_rest_server
serve(m3d_vton_rest_server.app, host='0.0.0.0', port=5000, url_scheme='https')

