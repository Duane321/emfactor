
emfactor <- function(x, k, iters = 100, verbose = FALSE){

  #Need to insert checks for the matrix x.
  #drop rows which are completely missing.

  d <- ncol(x)
  n <- nrow(x)

  if(is.null(colnames(x))){
    colnames(x) <- paste0('c', seq(ncol(x)))
  }

  if(is.null(rownames(x))){
    rownames(x) <- paste0('r', seq(nrow(x)))
  }

  missing_names <- apply(x, 1, function(x) names(which(is.na(x))))
  none_missing <- length(missing_names) == 0
  na_pos <- is.na(x)

  aux_func <- function(){

    #When this function is called, x's missing values must be
    #impuned with the SAME z's that will be referenced here.
    log_pz <- -(1/2) * (n * k * log(2 * pi) + sum(diag2(zizi_sum)))
    log_px <- -(n/2) * (d * log(2 * pi) + sum(log(psi)))

    #These are each of the terms that show up within the trace function within the vignette
    t1 <- diag2(xixi_sum)
    t2 <- -2 * colSums(x) * mu
    t3 <- n * mu * mu
    t4 <- -2 * diag2(xizi_sum %*% t(w))
    t5 <- 2 * mu*(w %*% colSums(z))
    t6 <- diag2(w %*% zizi_sum %*% t(w))

    log_px <- log_px - (1/2)*sum((t1 + t2 + t3 + t4 + t5 + t6)/psi)

    val <- log_pz + log_px
    aux_vals <<- c(aux_vals,val)
  }

  #initialization
  aux_vals <- c()
  z <- MASS::mvrnorm(n, rep(0, k), diag2(rep(1, k)))
  rownames(z) <- rownames(x)
  psi <- rep(NA, d)
  names(psi) <- colnames(x)
  mu <- colMeans(x, na.rm = TRUE)
  #To start, we fill in missing values with their column means
  if(any(na_pos)){
    for (cn in colnames(x)){
      x[is.na(x[, cn]), cn] <- mu[cn]
    }
  }
  xizi_sum <- t(x) %*% z
  xixi_sum <- t(x) %*% x
  zizi_sum <- diag2(rep(n, k)) + t(z) %*% z
  x_demeaned <- sweep(x, 2, mu)

  #avoid initialing psi and w, since that drives local solutions.

  #iterations
  for (t in seq(iters)){
    if(verbose){
      print(paste('iteration', t, '...'))
    }

    #MAXIMIZE
    mu <- colMeans(x)
    w <- (xizi_sum - outer(mu, colSums(z))) %*% solve(zizi_sum)
    scat <- (1/n)*(xixi_sum - n * outer(mu, mu))
    var_expl <- w %*% (zizi_sum/n) %*% t(w)
    for (nm in names(psi)){
      psi[nm] <- (scat[nm, nm] - var_expl[nm, nm])
    }
    psi <- pmax(psi, 0.00001)

    #EXPECTATION
    zizi_sum <- matrix(rep(0, k^2), ncol = k)
    xixi_sum <- matrix(rep(0, d^2), ncol = d, dimnames = list(colnames(x), colnames(x)))
    xizi_sum <- matrix(rep(0, d * k),ncol = k, dimnames = list(colnames(x)))
    g <- solve(diag2(rep(1, k)) + t(w) %*% sweep(w, 1, psi, "/"))
    z <- t(g %*% t(w) %*% sweep(t(x_demeaned), 1, psi, "/"))
    x_exp <- sweep(t(w %*% t(z)), 2, mu, '+')
    x[na_pos] <- x_exp[na_pos]
    x_demeaned <- sweep(x, 2, mu)
    for (i in seq(n)){
      zizi <- g + outer(z[i, ], z[i, ])
      xixi <- outer(x[i, ], x[i, ])
      xizi <- outer(x[i, ], z[i, ])

      zizi_eig <- eigen(zizi)
      zizi_sqrt <- zizi_eig$vectors %*% diag2(sqrt(zizi_eig$values))

      if(!none_missing){
        miss_n <- missing_names[[i]]
      } else{
        miss_n <- NULL
      }
      if(length(miss_n) > 0){
        obs_n <- setdiff(colnames(x), miss_n)
        if(length(miss_n) ==1 ){
          uncond_var <- w[miss_n, ] %*% zizi %*% w[miss_n, ] + psi[miss_n]
          vh_cov <- w[obs_n, ] %*% zizi %*% w[miss_n, ]
          vv_cov_inv <- invert_by_mil(psi[obs_n], w[obs_n,] %*% zizi_sqrt, t(zizi_sqrt) %*% t(w[obs_n, ]))
          cond_var <- uncond_var - t(vh_cov) %*% vv_cov_inv %*% vh_cov
        } else {
          uncond_var <- w[miss_n, ] %*% zizi %*% t(w[miss_n, ]) + diag2(psi[miss_n])
          if(length(obs_n) > 0){
            vh_cov <- w[obs_n, ] %*% zizi %*% t(w[miss_n, ])
            if(length(obs_n) == 1){
              vv_cov <- w[obs_n, ] %*% zizi %*% w[obs_n, ] + psi[obs_n]
              vv_cov_inv <- 1/vv_cov
            } else {
              vv_cov_inv <- invert_by_mil(psi[obs_n], w[obs_n,] %*% zizi_sqrt, t(zizi_sqrt) %*% t(w[obs_n, ]))
            }
            cond_var <- uncond_var - t(vh_cov) %*% vv_cov_inv %*% vh_cov
          } else {
            cond_var <- uncond_var
          }
        }
        xihxih <-  cond_var + outer(x[i, miss_n], x[i, miss_n])
        xixi[miss_n,miss_n] <- xihxih
        xizi[miss_n,] <- w[miss_n, ] %*% zizi + outer(mu[miss_n], z[i, ])
      }
      xizi_sum <- xizi_sum + xizi
      xixi_sum <- xixi_sum + xixi
      zizi_sum <- zizi_sum + zizi
    }

    aux_func()

    #stopping rule
    if(t == round(iters/3)){
      ave_val <- mean(abs(aux_vals))
      small <- ave_val * 0.001
    }
    if(t > round(iters/3)){
      diff_last <- aux_vals[length(aux_vals)] - aux_vals[length(aux_vals) - 1]
      if(diff_last < small){
        break
      }
    }
  }
  cov_est <- w %*% (zizi_sum/n) %*% t(w) + diag2(psi)
  out <- list(mu = mu, w = w, psi = psi, z = z, cov_est = cov_est)
  out

}
