norm <- function(x) sqrt(sum(x*x))

unitproj <- function(e,a){
  #Projection of vector a onto unit (assumption) vector e.
  sum(e * a) *e
}

project <- function(v1,v2){
  #project v2 onto v1
  (sum(v2 * v1)/sum(v1 * v1)) * v1
}

invert_by_mil <- function(l11,l21,l12){
  #This returns the inverse of a matrix that is diag(l11) + l21*l12
  #using the matrix inversion lemma. In this case, l11 must a
  #vector.
  small_d <- ncol(l21)
  mat <- solve(diag(rep(1,small_d)) + l12 %*% diag(1/l11) %*% l21)
  diag(1/l11) - diag(1/l11) %*% l21 %*% mat %*% l12 %*% diag(1/l11)
}

diag2 <- function(x){
  if(length(x)==1) {
    return(x)
  } else {
    return(diag(x))
  }
}

orthogonalize <- function(vecs,normalize=TRUE){
  #This function accepts a matrix and returns a matrix
  #with the same span, but with orthonormal columns.
  #This is done using the Gram–Schmidt process.

  vecs_out <- vecs
  vecs_out[] <- NA

  for(c in seq(ncol(vecs))){
    if(c==1) {
      if(normalize){
        vecs_out[,c] <- vecs[,c]/norm(vecs[,c])
      } else {
        vecs_out[,c] <- vecs[,c]
      }
    } else {
      vecs_outcurr <- vecs_out[,1:(c-1)]
      if(normalize){
        prjs <- plyr::laply(as.data.frame(vecs_outcurr),function(x) unitproj(x,vecs[,c]))
      } else{
        prjs <- plyr::laply(as.data.frame(vecs_outcurr),function(x) project(x,vecs[,c]))
      }
      if(c>2){
        prjs <- colSums(prjs)
      }
      vecs_out[,c] <- vecs[,c] - prjs
      if(normalize){
        vecs_out[,c] <- vecs_out[,c]/norm(vecs_out[,c])
      }
    }
  }
  vecs_out
}
