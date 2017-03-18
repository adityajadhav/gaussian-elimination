CC=gcc
compile: pthread_gauss.c
	$(CC) pthread_gauss.c -pthread -o pthread_gauss.out
run:
	@$(MAKE) && ./pthread_gauss.out $(msize) $(seed) $(output-file)
clean:
	rm -rf *.out
