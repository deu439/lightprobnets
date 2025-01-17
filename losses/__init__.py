import losses.classification_losses
import losses.endpoint_error
import losses.probabilistic_classification_losses
import losses.probabilistic_endpoint_error
import losses.elbo_loss
import losses.contrastive_loss
import losses.unsupervised_loss
import losses.elbo_loss_fb
import losses.unsupervised_loss_fb

ClassificationLoss = classification_losses.ClassificationLoss
DirichletProbOutLoss = probabilistic_classification_losses.DirichletProbOutLoss
MultiScaleEPE = endpoint_error.MultiScaleEPE
MultiScaleLaplacian = probabilistic_endpoint_error.MultiScaleLaplacian
EPE = endpoint_error.EPE
Elbo = elbo_loss.Elbo
MultiScaleElbo = elbo_loss.MultiScaleElbo
ContrastiveLoss = contrastive_loss.ContrastiveLoss
MultiScaleElboUpflow = elbo_loss.MultiScaleElboUpflow
Unsupervised = unsupervised_loss.Unsupervised
UnsupervisedFB = unsupervised_loss_fb.UnsupervisedFB
ElboFB = elbo_loss_fb.ElboFB
UnsupervisedSequence = unsupervised_loss.UnsupervisedSequence
UnsupervisedSequenceFB = unsupervised_loss_fb.UnsupervisedSequenceFB
