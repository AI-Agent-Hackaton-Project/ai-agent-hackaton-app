
ENV_DEV=.env.dev
ENV_PROD=.env.prod

dev-up:
	docker-compose -f docker-compose.dev.yml --env-file $(ENV_DEV) up --build

dev-down:
	docker-compose -f docker-compose.dev.yml down

prod-up:
	docker-compose -f docker-compose.prod.yml --env-file $(ENV_PROD) up --build

prod-down:
	docker-compose -f docker-compose.prod.yml down

down:
	docker-compose -f docker-compose.dev.yml down
	docker-compose -f docker-compose.prod.yml down
