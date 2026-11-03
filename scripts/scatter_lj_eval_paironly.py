"""scatter_lj_eval with the mesh branch switched off at inference (pair term only), same args as scatter_lj_eval."""
from scripts import scatter_lj as sl
from scripts import scatter_lj_diag as d
from scripts import scatter_lj_eval as ev

sl.PeriodicTiles = lambda model, knn=sl.KNN: d.OptTiles(model, knn, use_mesh=False)

if __name__ == "__main__":
    ev.main()
