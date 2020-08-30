import socket

mysock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#AF_INET is actually address from the internet which takes 2 arguments 1)URL 2)port number (int)
#SOCK_STREAM is used to create TCP protocols
mysock.connect(('data.pr4e.org',80))
cmd='GET http://data.pr4e.org/romeo.txt HTTP/1.0 \n\n'.encode()
mysock.send(cmd)

while True:
    data=mysock.recv(512)
    if(len(data)<1):
        break
    print(data.decode())
mysock.close()

