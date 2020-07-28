 #!/bin/bash
 app="pvpanels"
 docker build -t ${app} .
 docker run -d -e PYTHONUNBUFFERED=1 -p 5000:5000 \
   --name=${app} \
   -v $PWD:/app ${app}
